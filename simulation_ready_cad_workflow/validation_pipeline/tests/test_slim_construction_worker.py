"""Automatic setup exclusions expire with their owner; manual pauses persist."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from ..slim_harness.construction_budget import ProjectBudget,initialize_budget
from ..slim_harness.construction_worker import _simulation_command, communicate_budgeted, shell_execution_events


PREFIX='simulation_ready_cad_workflow.validation_pipeline.slim_harness.'


def _event(identifier,kind='item.started'):
    return {'type':kind,'item':{'id':identifier,'type':'command_execution','command':'uv pip install fixture-package'}}


def _append(path,event):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event)+'\n')


def _crash_install_monitor(project,events,checkpoint):
    class CrashingProcess:
        args=['isolated-crash-fixture']
        def communicate(self,**kwargs):
            ledger=json.loads((Path(project)/'construction_budget.json').read_text())
            Path(checkpoint).write_text(json.dumps({'pause':ledger['pause'],
                'session_kinds':[row['kind'] for row in ledger['sessions'].values()]}))
            # Unlike raising an exception, this cannot execute ExitStack or
            # activity finally blocks. It models abrupt supervisor termination.
            os._exit(23)
    communicate_budgeted(CrashingProcess(),'fixture',100,project,events)


class ConstructionWorkerExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.project=Path(self.temp.name);self.events=self.project/'events.jsonl';self.now=1000.
        clock=patch(PREFIX+'construction_budget.time.monotonic',side_effect=lambda:self.now)
        clock.start();self.addCleanup(clock.stop)
        boot=patch(PREFIX+'construction_budget._boot_id',return_value='test-boot')
        boot.start();self.addCleanup(boot.stop)
        heartbeat=patch(PREFIX+'construction_budget._Activity.start')
        heartbeat.start();self.addCleanup(heartbeat.stop)
        initialize_budget(self.project)
        self.budget=ProjectBudget(self.project,lease_seconds=10000)
        selected=patch(PREFIX+'construction_worker.ProjectBudget',return_value=self.budget)
        selected.start();self.addCleanup(selected.stop)

    def test_overlapping_installs_exclude_until_last_completion_then_charge_design(self):
        _append(self.events,_event('install-a'));_append(self.events,_event('install-b'))
        calls=[]
        process=Mock(args=['fixture'])
        def communicate(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                self.now+=10;_append(self.events,_event('install-a','item.completed'))
                raise subprocess.TimeoutExpired(process.args,.5)
            if len(calls)==2:
                self.now+=5;_append(self.events,_event('install-b','item.completed'))
                raise subprocess.TimeoutExpired(process.args,.5)
            self.now+=4
            return b'',b''
        process.communicate.side_effect=communicate
        communicate_budgeted(process,'prompt',100,self.project,self.events)
        snapshot=self.budget.snapshot()
        self.assertEqual(snapshot['excluded_seconds'],15)
        self.assertEqual(snapshot['active_seconds'],4)
        self.assertFalse(snapshot['paused'])
        self.assertEqual(snapshot['active_sessions'],0)
        self.assertEqual(calls[0]['input'],'prompt')
        self.assertIsNone(calls[1]['input'])

    def test_cursor_shell_installs_are_excluded_until_completion(self):
        self.assertEqual(
            [('started','install-a','uv pip install fixture-package')],
            shell_execution_events({
                'type':'tool_call','subtype':'started','call_id':'install-a',
                'tool_call':{'shellToolCall':{'args':{'command':'uv pip install fixture-package'}}},
            }),
        )
        _append(self.events,{
            'type':'tool_call','subtype':'started','call_id':'install-a',
            'tool_call':{'function':{'name':'Shell','arguments':json.dumps({'command':'uv pip install fixture-package'})}},
        })
        calls=[]
        process=Mock(args=['fixture'])
        def communicate(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                self.now+=10
                _append(self.events,{
                    'type':'tool_call','subtype':'completed','call_id':'install-a',
                    'tool_call':{'function':{'name':'Shell','arguments':json.dumps({'command':'uv pip install fixture-package'})}},
                })
                raise subprocess.TimeoutExpired(process.args,.5)
            self.now+=4
            return b'',b''
        process.communicate.side_effect=communicate
        communicate_budgeted(process,'prompt',100,self.project,self.events)
        snapshot=self.budget.snapshot()
        self.assertEqual(snapshot['excluded_seconds'],10)
        self.assertEqual(snapshot['active_seconds'],4)
        self.assertFalse(snapshot['paused'])

    def test_simulation_shell_command_is_excluded_only_until_completion(self):
        command=('PYTHONPATH=/repo python -X utf8 -m '
                 'simulation_ready_cad_workflow.validation_pipeline.slim_harness '
                 '--workspace /repo simulate scene.json --test move')
        _append(self.events,{'type':'item.started','item':{'id':'sim-a','type':'command_execution','command':command}})
        process=Mock(args=['fixture']);calls=[]
        def communicate(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                self.now+=10
                _append(self.events,{'type':'item.completed','item':{'id':'sim-a','type':'command_execution','command':command}})
                raise subprocess.TimeoutExpired(process.args,.5)
            self.now+=4
            return b'',b''
        process.communicate.side_effect=communicate
        communicate_budgeted(process,'prompt',100,self.project,self.events)
        snapshot=self.budget.snapshot()
        self.assertEqual(snapshot['excluded_by_reason'],{'simulation_wait':10})
        self.assertEqual(snapshot['active_seconds'],4)
        self.assertEqual(snapshot['active_sessions'],0)

    def test_claude_shell_events_bound_a_simulation_exclusion(self):
        command=('python -m '
                 'simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime '
                 'runs/r01/request.json')
        started={'type':'assistant','message':{'content':[
            {'type':'text','text':'Running the development simulation.'},
            {'type':'tool_use','id':'toolu_sim','name':'Bash','input':{'command':command}}]}}
        self.assertEqual([('started','toolu_sim',command)],shell_execution_events(started))
        self.assertEqual([],shell_execution_events({'type':'assistant','message':{'content':[
            {'type':'tool_use','id':'toolu_read','name':'Read','input':{'file_path':'/x'}}]}}))
        _append(self.events,started)
        process=Mock(args=['fixture']);calls=[]
        def communicate(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                self.now+=10
                _append(self.events,{'type':'user','message':{'content':[
                    {'type':'tool_result','tool_use_id':'toolu_sim','content':'done'}]}})
                raise subprocess.TimeoutExpired(process.args,.5)
            self.now+=4
            return b'',b''
        process.communicate.side_effect=communicate
        communicate_budgeted(process,'prompt',100,self.project,self.events)
        snapshot=self.budget.snapshot()
        self.assertEqual(snapshot['excluded_by_reason'],{'simulation_wait':10})
        self.assertEqual(snapshot['active_seconds'],4)

    def test_batched_claude_tool_results_close_their_shell_exclusions(self):
        command=('python -m '
                 'simulation_ready_cad_workflow.validation_pipeline.slim_harness '
                 '--workspace /w simulate scene.json')
        _append(self.events,{'type':'assistant','message':{'content':[
            {'type':'tool_use','id':'toolu_read','name':'Read','input':{'file_path':'/x'}},
            {'type':'tool_use','id':'toolu_sim','name':'Bash','input':{'command':command}},
            {'type':'tool_use','id':'toolu_install','name':'Bash','input':{'command':'uv pip install fixture-package'}}]}})
        process=Mock(args=['fixture']);calls=[]
        def communicate(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                self.now+=10
                _append(self.events,{'type':'user','message':{'content':[
                    {'type':'tool_result','tool_use_id':'toolu_read','content':'file'},
                    {'type':'tool_result','tool_use_id':'toolu_sim','content':'done'},
                    {'type':'tool_result','tool_use_id':'toolu_install','content':'installed'}]}})
                raise subprocess.TimeoutExpired(process.args,.5)
            self.now+=30
            return b'',b''
        process.communicate.side_effect=communicate
        communicate_budgeted(process,'prompt',100,self.project,self.events)
        snapshot=self.budget.snapshot()
        # Both exclusions closed with their shell calls; the overlap is counted
        # once, under the alphabetically first reason.
        self.assertEqual(snapshot['excluded_seconds'],10)
        self.assertEqual(snapshot['active_seconds'],30)
        self.assertEqual(snapshot['active_sessions'],0)

    def test_simulation_command_detection(self):
        module='simulation_ready_cad_workflow.validation_pipeline.slim_harness'
        self.assertTrue(_simulation_command(f'python -m {module} simulate scene.json'))
        self.assertTrue(_simulation_command(f'PYTHONPATH=/x python -X utf8 -m {module} --workspace /w probe scene.json --test a'))
        self.assertTrue(_simulation_command(f'python -m {module} --author-visual-policy=full simulate scene.json'))
        self.assertTrue(_simulation_command(f'python -m {module}.runtime runs/r01/request.json'))
        self.assertTrue(_simulation_command(f'python -m {module}.runtime r.json && echo done'))
        self.assertFalse(_simulation_command(f'python -m {module} budget status --project p'))
        self.assertFalse(_simulation_command(f'python -m {module} create --project p --brief "simulate a latch"'))
        self.assertFalse(_simulation_command(f'python -m {module} submit scene.json --project p'))
        self.assertFalse(_simulation_command(f'python -m {module}_extra simulate scene.json'))
        self.assertFalse(_simulation_command('python scene.py'))
        self.assertFalse(_simulation_command(None))

    def test_manual_pause_started_during_auto_install_is_not_resumed_by_completion(self):
        _append(self.events,_event('install'))
        process=Mock(args=['fixture']);calls=[]
        def communicate(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                self.now+=2
                self.budget.pause('human_intervention')
                self.now+=3;_append(self.events,_event('install','item.completed'))
                raise subprocess.TimeoutExpired(process.args,.5)
            self.now+=4
            return b'',b''
        process.communicate.side_effect=communicate
        communicate_budgeted(process,'prompt',100,self.project,self.events)
        snapshot=self.budget.snapshot()
        self.assertTrue(snapshot['paused'])
        self.assertEqual(snapshot['pause']['reason'],'human_intervention')
        self.assertEqual(snapshot['excluded_by_reason'],{'environment_setup':2,'human_intervention':7})
        self.assertEqual(snapshot['active_seconds'],0)

    def test_preexisting_manual_pause_is_never_replaced_by_an_installer(self):
        self.budget.pause('human_intervention');_append(self.events,_event('install'))
        process=Mock(args=['fixture'])
        def communicate(**kwargs):
            self.now+=2
            return b'',b''
        process.communicate.side_effect=communicate
        communicate_budgeted(process,'prompt',100,self.project,self.events)
        snapshot=self.budget.snapshot()
        self.assertEqual(snapshot['pause']['reason'],'human_intervention')
        self.assertEqual(snapshot['excluded_by_reason'],{'human_intervention':2})
        self.assertEqual(snapshot['active_sessions'],0)

    def test_stale_auto_exclusion_lease_cannot_leave_future_work_unmetered(self):
        setup=ProjectBudget(self.project,lease_seconds=5)
        with self.budget.activity('workflow_supervisor',heartbeat=False):
            context=setup.activity('environment_setup',heartbeat=False);context.__enter__()
            self.now+=20
            # When a sandbox cannot inspect the other owner's PID, the lease
            # alone must still expire this automatic exclusion.
            with patch(PREFIX+'construction_budget._owner_alive',return_value=True):
                snapshot=self.budget.snapshot()
            self.assertEqual(snapshot['excluded_seconds'],5)
            self.assertEqual(snapshot['active_seconds'],15)
            self.assertFalse(snapshot['paused'])
            context.__exit__(None,None,None)


class ConstructionWorkerHardCrashTests(unittest.TestCase):
    def test_hard_exit_leaves_no_global_pause_and_next_worker_is_charged(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);budget=initialize_budget(project)
            events=project/'events.jsonl';checkpoint=project/'crash-checkpoint.json'
            _append(events,_event('install'))
            process=multiprocessing.get_context('spawn').Process(target=_crash_install_monitor,
                args=(str(project),str(events),str(checkpoint)))
            try:
                process.start();process.join(timeout=10)
                self.assertEqual(process.exitcode,23)
                observed=json.loads(checkpoint.read_text())
                self.assertIsNone(observed['pause'])
                self.assertIn('environment_setup',observed['session_kinds'])
                snapshot=budget.snapshot()
                self.assertFalse(snapshot['paused'])
                if os.name!='nt':self.assertEqual(snapshot['active_sessions'],0)
                # On Windows PID liveness is intentionally unknown; advance
                # beyond the default lease in the observer only.
                later=time.monotonic()+budget.lease_seconds+1
                with patch(PREFIX+'construction_budget.time.monotonic',side_effect=lambda:later):
                    before=budget.snapshot()
                    with budget.activity('agent',heartbeat=False):
                        later+=1
                    after=budget.snapshot()
                self.assertAlmostEqual(after['active_seconds']-before['active_seconds'],1,places=5)
                self.assertEqual(after['excluded_seconds'],before['excluded_seconds'])
            finally:
                if process.is_alive():process.terminate();process.join(timeout=5)


if __name__=='__main__':unittest.main()
