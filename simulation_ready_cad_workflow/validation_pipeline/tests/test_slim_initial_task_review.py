"""Initial independent task review; no model calls or mechanical experiments."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ..slim_harness.review import CRITERIA_SCHEMA, make_criteria


class Worker:
    def __init__(self, response):self.response=response;self.calls=[]
    def run_turn(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        return deepcopy(self.response)


class InitialTaskReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.task=self.root/'task.md';self.output=self.root/'criteria.json'
        self.original='Fixture machine: move both ways and hold under load for 999 seconds.'
        self.adjusted='Fixture machine: move both ways and hold under load for 2 seconds.'
        self.task.write_text(self.original,encoding='utf-8')
        self.tier={'name':'light','active_budget_seconds':3600,'simulation_timeout_seconds':60,
                   'max_tasks_per_design':2,'min_parts':0,'max_parts':10}

    def response(self,task=None,*,review=False,status='ready'):
        task=task or self.original
        value={'status':status,'summary':'Independent fixture review',
               'criteria':[{'id':'motion','kind':'function','statement':'Move both ways',
                            'source_quote':'move both ways','verification':'Observe both directions'},
                           {'id':'load','kind':'function','statement':'Hold under load',
                            'source_quote':task.split(' and ')[1].rstrip('.'),
                            'verification':'Observe the stated loaded hold'}]}
        if review:value.update(task_text=task,task_review={'reasonable':status=='ready','reason':'Both fixture functions retained within the declared tier'})
        return value

    def run_review(self,worker,**kwargs):
        return make_criteria(worker,self.root,self.task,self.root/'turn',self.output,**kwargs)

    def test_default_keeps_original_schema_and_persists_unchanged_task_criteria(self):
        worker=Worker(self.response());result=self.run_review(worker)
        self.assertEqual('ready',result['status'])
        saved=json.loads(self.output.read_text())
        self.assertEqual('slim_frozen_criteria_v1',saved['schema_version'])
        self.assertEqual(self.original,saved['task_text'])
        self.assertEqual(self.original,self.task.read_text())
        self.assertIs(CRITERIA_SCHEMA,worker.calls[0][1]['response_schema'])
        self.assertEqual(1,len(worker.calls))

    def test_adjustable_review_is_one_fresh_readonly_call_and_does_not_apply_text(self):
        worker=Worker(self.response(self.adjusted,review=True))
        result=self.run_review(worker,task_adjustable=True,tier_config=self.tier)
        self.assertEqual(self.adjusted,result['task_text'])
        self.assertTrue(result['task_review']['reasonable'])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.original,self.task.read_text())
        self.assertEqual(1,len(worker.calls))
        args,kwargs=worker.calls[0]
        self.assertTrue(kwargs['readonly']);self.assertTrue(kwargs['fresh'])
        self.assertEqual(1200,args[3])
        self.assertIn(json.dumps(self.tier,ensure_ascii=False),args[1])
        self.assertIn('before any design exists',args[1])
        self.assertIn('Do not supply a mechanism architecture',args[1])
        self.assertEqual(['ready','needs_revision','unsupported'],kwargs['response_schema']['properties']['status']['enum'])
        self.assertEqual(['ready'],CRITERIA_SCHEMA['properties']['status']['enum'])

    def test_unchanged_adjustable_task_still_returns_a_recorded_review(self):
        worker=Worker(self.response(review=True))
        result=self.run_review(worker,task_adjustable=True,tier_config=self.tier)
        self.assertEqual(self.original,result['task_text'])
        self.assertTrue(result['task_review']['reason'])
        self.assertFalse(self.output.exists())

    def test_nonready_task_assessment_is_returned_without_freezing_or_provider_relabel(self):
        for status in ('needs_revision','unsupported','provider_error'):
            with self.subTest(status=status):
                response=self.response(review=True,status=status)
                response['criteria']=[]
                worker=Worker(response)
                result=self.run_review(worker,task_adjustable=True,tier_config=self.tier)
                self.assertEqual(status,result['status'])
                self.assertFalse(self.output.exists())
                self.assertEqual(self.original,self.task.read_text())

    def test_adjusted_criteria_must_quote_the_reviewed_text_not_old_metrics(self):
        response=self.response(review=True)
        response['task_text']=self.adjusted
        with self.assertRaisesRegex(ValueError,'unsupported source quotes'):
            self.run_review(Worker(response),task_adjustable=True,tier_config=self.tier)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.original,self.task.read_text())

    def test_ready_needs_reasonable_task_and_reason(self):
        for field in ('reasonable','reason'):
            with self.subTest(field=field):
                response=self.response(review=True)
                response['task_review'][field]=False if field=='reasonable' else ''
                with self.assertRaisesRegex(ValueError,'recorded reason'):
                    self.run_review(Worker(response),task_adjustable=True,tier_config=self.tier)
        self.assertFalse(self.output.exists())

    def test_a_tier_alone_does_not_enable_task_rewriting(self):
        worker=Worker(self.response())
        self.run_review(worker,tier_config=self.tier)
        self.assertIs(CRITERIA_SCHEMA,worker.calls[0][1]['response_schema'])
        self.assertEqual(self.original,json.loads(self.output.read_text())['task_text'])


if __name__=='__main__':unittest.main()
