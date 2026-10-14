"""Real short subprocess timing/kill; no model or mechanical experiment."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

import psutil

from ..cold_start.worker import CodexCliWorker
from ..slim_harness import task_change_review as module
from ..slim_harness.budget import ProjectBudget,TierConfig
from ..slim_harness.cli import create_project


CHILD=r'''
import json,os,subprocess,sys,time
from pathlib import Path
response=Path(sys.argv[1]);sid=sys.argv[2];behavior=sys.argv[3];second=sys.argv[4]=='second'
text=sys.stdin.read();(response.parent/'received-prompt.txt').write_text(text,encoding='utf-8')
if behavior!='no_session':print(json.dumps({'type':'thread.started','thread_id':sid}),flush=True)
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'Partial review: the changed test may omit a required condition.'}}),flush=True)
if behavior=='provider_error':
    print(json.dumps({'type':'turn.failed','error':{'message':'usage limit fixture'}}),flush=True)
    raise SystemExit(1)
if second and behavior=='finish':
    response.write_text(json.dumps({'conclusion':'I disagree with this change; retain the original condition.','issues':['Missing required condition'],'evidence_paths':[]}),encoding='utf-8')
    print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}),flush=True)
else:
    response.write_text('{"conclusion":"Unfinished raw conclusion',encoding='utf-8')
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (response.parent/'owned-processes.json').write_text(json.dumps({'parent':os.getpid(),'child':child.pid}))
    time.sleep(20)
'''


class ImmediateReviewer(CodexCliWorker):
    def __init__(self):super().__init__();self.calls=[]
    def run_turn(self,workspace,prompt,directory,timeout_seconds,**kwargs):
        self.calls.append((prompt,timeout_seconds,kwargs))
        Path(directory).mkdir(parents=True)
        result={'conclusion':'Fixture review','issues':[],'evidence_paths':[],
                'exit_code':0,'observed_session_id':'independent-fixture','elapsed_seconds':0.}
        (Path(directory)/'response.json').write_text(json.dumps(result),encoding='utf-8')
        return result


class TaskChangeReviewTests(unittest.TestCase):
    def setUp(self):
        workspace=Path(__file__).resolve().parents[3]
        base=workspace/'.runtime_runs/task_change_review_timer_20260913';base.mkdir(parents=True,exist_ok=True)
        self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'_',dir=base))
        self.child=self.root/'fake_cli.py';self.child.write_text(CHILD,encoding='utf-8')

    def execute(self,*,tier='light',behavior='hang',budget=None,soft=None,grace=.7):
        worker=CodexCliWorker(session_id='author-session-must-not-change')
        commands=[];actual_popen=subprocess.Popen
        def launch(command,**kwargs):
            commands.append(command)
            second='resume' in command
            sid=command[command.index('resume')+1] if second else 'independent-review-fixture'
            response=command[command.index('-o')+1]
            return actual_popen([sys.executable,'-X','utf8',str(self.child),response,sid,behavior,'second' if second else 'first'],**kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch('simulation_ready_cad_workflow.validation_pipeline.cold_start.worker.subprocess.Popen',side_effect=launch))
            stack.enter_context(patch.object(module,'REVIEW_SOFT_SECONDS',soft or {'light':.5,'medium':.65,'high':.8}))
            stack.enter_context(patch.object(module,'REVIEW_GRACE_SECONDS',grace))
            result=module.run_change_review(worker,self.root,'Assess only these submitted simulation-test changes.',self.root/'review',tier_name=tier,budget=budget)
        self.assertEqual('author-session-must-not-change',worker.session_id)
        (self.root/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
        return result,commands

    def assert_stopped(self,phase):
        self.assertTrue(phase['termination']['cleanup_confirmed'],phase)
        self.assertFalse(psutil.pid_exists(phase['process_pid']))
        children=json.loads((Path(phase['directory'])/'owned-processes.json').read_text())
        self.assertFalse(psutil.pid_exists(children['child']))

    def test_policy_caps_for_all_tiers_ignore_simulation_timeout_and_none_defaults_light(self):
        for tier,expected in (('light',60.),('medium',120.),('high',180.),(None,60.)):
            with self.subTest(tier=tier):
                workspace=self.root/str(tier);project=create_project(workspace,'fixture','Task changes')
                budget=ProjectBudget(project,TierConfig('test-policy',1000,.01,2,0,10))
                budget.limit=Mock(side_effect=AssertionError('Physical limit must not cap review'))
                reviewer=ImmediateReviewer()
                result=module.run_change_review(reviewer,workspace,'Only changed tests',workspace/'review',tier_name=tier,budget=budget)
                self.assertEqual('completed',result['status'])
                self.assertEqual(expected,result['soft_seconds'])
                self.assertEqual(60.,result['hard_after_soft_seconds'])
                self.assertGreater(reviewer.calls[0][1],expected-1.)
                self.assertLessEqual(reviewer.calls[0][1],expected)
                self.assertIn(f'最多审核 {expected/60:g} min',reviewer.calls[0][0])
                self.assertEqual(0,result['reminder_count'])

    def test_soft_stop_and_reminder_finish_in_same_independent_session_for_each_tier(self):
        for tier in ('light','medium','high'):
            with self.subTest(tier=tier):
                oldroot=self.root
                self.root=oldroot/tier;self.root.mkdir()
                try:
                    result,commands=self.execute(tier=tier,behavior='finish')
                    self.assertEqual('completed',result['status'],result)
                    self.assertEqual(1,result['reminder_count'])
                    self.assertEqual(2,len(commands))
                    self.assertNotIn('resume',commands[0]);self.assertNotIn('--ephemeral',commands[0])
                    self.assertEqual('independent-review-fixture',commands[1][commands[1].index('resume')+1])
                    self.assertTrue(all(command[command.index('--sandbox')+1]=='read-only' for command in commands))
                    self.assert_stopped(result['phases'][0])
                    self.assertEqual(module.TIME_REMINDER,(self.root/'review/reminder/received-prompt.txt').read_text(encoding='utf-8'))
                    self.assertIn('I disagree',result['raw_output'])
                    self.assertIn('Unfinished raw conclusion',result['raw_output'])
                    self.assertFalse(result['review_is_acceptance_gate'])
                finally:self.root=oldroot

    def test_hard_deadline_kills_only_review_process_trees_and_retains_partial_text(self):
        sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            result,commands=self.execute()
            self.assertEqual('partial',result['status'])
            self.assertEqual('hard_deadline',result['stop_reason'])
            self.assertEqual(1,result['reminder_count']);self.assertEqual(2,len(commands))
            for phase in result['phases']:self.assert_stopped(phase)
            self.assertIn('Partial review:',result['raw_output'])
            self.assertIn('Unfinished raw conclusion',result['raw_output'])
            self.assertLess(result['wall_seconds'],2.5)
            self.assertIsNone(sentinel.poll())
            (self.root/'unrelated-process.json').write_text(json.dumps({'pid':sentinel.pid,'alive_after_review_kills':True}))
        finally:
            if sentinel.poll() is None:sentinel.kill()
            sentinel.wait(timeout=2)

    def test_project_budget_preempts_soft_and_never_adds_a_reminder_or_credit(self):
        project=create_project(self.root,'budget','Task changes')
        budget=ProjectBudget(project,'light');budget.tracker._active_seconds=3599.7;budget.save()
        result,commands=self.execute(budget=budget)
        self.assertEqual('project_budget_exhausted',result['stop_reason'],result)
        self.assertEqual(0,result['reminder_count']);self.assertEqual(1,len(commands))
        self.assertLessEqual(result['phases'][0]['timeout_seconds'],.3)
        checkpoint=json.loads((project/'budget_checkpoint.json').read_text())
        self.assertEqual(3600,checkpoint['budget_limit_seconds'])
        self.assertEqual(0,checkpoint['remaining_active_seconds'])
        self.assertFalse((project/'budget_grants.json').exists())

    def test_no_session_does_not_start_another_review_and_provider_error_is_not_retried(self):
        for behavior,reason in (('no_session','independent_session_unavailable_no_restart'),('provider_error','review_failed_without_retry')):
            with self.subTest(behavior=behavior):
                oldroot=self.root;self.root=oldroot/behavior;self.root.mkdir()
                try:
                    result,commands=self.execute(behavior=behavior)
                    self.assertEqual(reason,result['stop_reason'],result)
                    self.assertEqual(1,len(commands));self.assertEqual(0,result['reminder_count'])
                    self.assertTrue(result['raw_output'])
                finally:self.root=oldroot

    def test_unavailable_reviewer_and_unknown_tier_return_durable_partial_advice(self):
        for index,(reviewer,tier) in enumerate(((object(),'light'),(ImmediateReviewer(),'unknown'))):
            out=self.root/str(index)
            result=module.run_change_review(reviewer,self.root,'changes',out,tier_name=tier)
            self.assertEqual('partial',result['status'])
            self.assertTrue(result['raw_output'])
            self.assertEqual([],result['phases'])
            self.assertEqual(result,json.loads((out/'review-result.json').read_text(encoding='utf-8')))


if __name__=='__main__':unittest.main()
