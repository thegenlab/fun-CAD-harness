"""Audit resource bounds and incomplete evidence, without launching real CAD."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import BoundedSubprocessError,BoundedSubprocessResult
from ..slim_harness.construction_audit import execute_motion_audit,run_motion_audit_bounded
from ..slim_harness.construction_limits import ConstructionLimitError


PREFIX='simulation_ready_cad_workflow.validation_pipeline.slim_harness.'


class ConstructionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.prepared=self.root/'prepared-scene.json';self.trace=self.root/'physical.trace.json'
        self.output=self.root/'geometry-audit.json'
        write_json_atomic(self.prepared,{'bodies':[],'validation_scope':{'mode':'equivalent_elastic'}})
        write_json_atomic(self.trace,{'test_id':'load','physical_pass':True,
                                    'samples':[{'time_s':0},{'time_s':.001},{'time_s':.002}],'signals':{}})
        self.before=(self.prepared.read_bytes(),self.trace.read_bytes())
        self.audit={'schema_version':'slim_geometry_audit_v1','passed':True,
                    'available_solver_samples':3,'checked_solver_samples':3,'violations':[]}

    def run_audit(self,remaining=1200,policy=None):
        return run_motion_audit_bounded(self.prepared,self.trace,self.output,
            policy={'simulation_timeout_seconds':180} if policy is None else policy,remaining_seconds=remaining,
            cwd=self.root,cad_python='/fixture/cad-python',test_id='load')

    def check_inputs_unchanged(self):
        self.assertEqual(self.before,(self.prepared.read_bytes(),self.trace.read_bytes()))

    def test_success_uses_full_child_result_with_remaining_work_wall_and_memory_bounds(self):
        def child(command,**kwargs):
            self.assertEqual(command[0],'/fixture/cad-python')
            self.assertEqual(command[-3:-1],[str(self.prepared),str(self.trace)])
            self.assertNotEqual(Path(command[-1]),self.output)
            self.assertEqual(kwargs['timeout_seconds'],73)
            self.assertEqual(kwargs['max_rss_bytes'],'auto')
            write_json_atomic(Path(command[-1]),self.audit)
            return BoundedSubprocessResult(0,b'audit output',b'')
        with patch(PREFIX+'construction_audit.run_bounded_subprocess',side_effect=child):
            actual=self.run_audit(73)
        self.assertEqual(actual,self.audit)
        self.assertEqual(json.loads(self.output.read_text()),self.audit)
        cost=json.loads((self.root/'construction-audit-cost.json').read_text())
        self.assertEqual(cost['status'],'finished')
        self.assertEqual(cost['effective_timeout_seconds'],73)
        self.assertIn('not a combined',cost['scope'])
        self.check_inputs_unchanged()

    def test_timeout_preserves_physics_and_replaces_stale_audit_pass_with_incomplete(self):
        write_json_atomic(self.output,self.audit)
        with patch(PREFIX+'construction_audit.run_bounded_subprocess',
                   side_effect=BoundedSubprocessError('BOUNDED_SUBPROCESS_TIMEOUT','fixture wall deadline')) as launch:
            with self.assertRaises(ConstructionLimitError) as caught:self.run_audit()
        self.assertEqual(launch.call_args.kwargs['timeout_seconds'],600)
        self.assertEqual(caught.exception.code,'CONSTRUCTION_AUDIT_HARD_TIMEOUT')
        self.assertEqual(caught.exception.details['evidence_status'],'incomplete')
        self.assertEqual(len(caught.exception.details['revision_strategies']),3)
        output=json.loads(self.output.read_text())
        self.assertFalse(output['passed'])
        self.assertEqual(output['status'],'evidence_incomplete')
        self.assertIsNone(output['checked_solver_samples'])
        self.assertEqual(json.loads((self.root/'construction-audit-cost.json').read_text())['status'],'timeout')
        self.check_inputs_unchanged()

    def test_explicit_audit_cap_and_remaining_budget_work_for_each_policy_shape(self):
        from ..slim_harness.construction_budget import BudgetConfig
        policies=[{'simulation_timeout_seconds':20,'audit_timeout_seconds':180},
                  {'config':{'simulation_timeout_seconds':20,'audit_timeout_seconds':180}},
                  BudgetConfig(simulation_timeout_seconds=20,audit_timeout_seconds=180)]
        for policy in policies:
            for remaining,expected in [(1200,180),(42,42)]:
                with self.subTest(policy=policy,remaining=remaining):
                    def child(command,**kwargs):
                        self.assertEqual(kwargs['timeout_seconds'],expected)
                        write_json_atomic(Path(command[-1]),self.audit)
                        return BoundedSubprocessResult(0,b'',b'')
                    with patch(PREFIX+'construction_audit.run_bounded_subprocess',side_effect=child):
                        self.run_audit(remaining,policy)
                    cost=json.loads((self.root/'construction-audit-cost.json').read_text())
                    self.assertEqual(cost['configured_audit_timeout_seconds'],180)
                    self.assertEqual(cost['native_simulation_timeout_seconds'],20)

    def test_longer_audit_cap_does_not_increase_native_simulation_timeout(self):
        from ..slim_harness.construction_budget import BudgetConfig
        from ..slim_harness.construction_limits import run_simulation_bounded
        test={'id':'load','duration_s':1}
        # Without extensions, so the native limit is exactly the configured one.
        with patch(PREFIX+'construction_limits.run_bounded_subprocess',return_value=BoundedSubprocessResult(0,b'',b'')) as launch, \
                patch(PREFIX+'construction_limits.SIMULATION_EXTENSION_CEILING_SECONDS',0.):
            run_simulation_bounded(['/fixture/chrono'],scene={'tests':[test]},geometry={'bodies':[]},test=test,
                policy=BudgetConfig(audit_timeout_seconds=600),remaining_seconds=1200,cwd=self.root,env={})
        self.assertEqual(launch.call_args.kwargs['timeout_seconds'],180)

    def test_audit_limit_must_be_finite_positive_for_all_policy_forms(self):
        from ..slim_harness.construction_limits import ConstructionLimits
        for policy in [{'audit_timeout_seconds':0},{'config':{'audit_timeout_seconds':-1}},
                       ConstructionLimits(audit_timeout_seconds=float('nan'))]:
            with self.subTest(policy=policy),patch(PREFIX+'construction_audit.run_bounded_subprocess') as launch:
                with self.assertRaises(ValueError):self.run_audit(policy=policy)
                launch.assert_not_called()

    def test_memory_limit_also_returns_incomplete_evidence_and_cost(self):
        with patch(PREFIX+'construction_audit.run_bounded_subprocess',
                   side_effect=BoundedSubprocessError('BOUNDED_SUBPROCESS_MEMORY_LIMIT','fixture RSS limit')):
            with self.assertRaises(ConstructionLimitError) as caught:self.run_audit()
        self.assertEqual(caught.exception.code,'CONSTRUCTION_AUDIT_RESOURCE_LIMIT')
        self.assertFalse(json.loads(self.output.read_text())['passed'])
        self.assertEqual(json.loads((self.root/'construction-audit-cost.json').read_text())['status'],'resource_error')
        self.check_inputs_unchanged()

    def test_truncated_success_cannot_become_geometry_pass(self):
        def child(command,**kwargs):
            write_json_atomic(Path(command[-1]),{**self.audit,'checked_solver_samples':2})
            return BoundedSubprocessResult(0,b'',b'')
        with patch(PREFIX+'construction_audit.run_bounded_subprocess',side_effect=child):
            with self.assertRaises(ConstructionLimitError) as caught:self.run_audit()
        self.assertEqual(caught.exception.code,'CONSTRUCTION_AUDIT_INCOMPLETE')
        self.assertFalse(json.loads(self.output.read_text())['passed'])

    def test_real_geometry_failure_is_returned_as_failure_not_timeout(self):
        failure={**self.audit,'passed':False,'checked_solver_samples':1,
                 'violations':[{'kind':'original_solid_interpenetration'}]}
        def child(command,**kwargs):
            write_json_atomic(Path(command[-1]),failure)
            return BoundedSubprocessResult(0,b'',b'')
        with patch(PREFIX+'construction_audit.run_bounded_subprocess',side_effect=child):
            self.assertEqual(self.run_audit(),failure)
        self.assertEqual(json.loads((self.root/'construction-audit-cost.json').read_text())['status'],'finished')

    def test_exhausted_project_does_not_start_an_audit(self):
        with patch(PREFIX+'construction_audit.run_bounded_subprocess') as launch:
            with self.assertRaises(ConstructionLimitError) as caught:self.run_audit(0)
        self.assertEqual(caught.exception.code,'PROJECT_BUDGET_EXHAUSTED')
        launch.assert_not_called();self.check_inputs_unchanged()

    def test_child_receives_every_stored_solver_sample_and_preserves_scope(self):
        prepared=json.loads(self.prepared.read_text());trace=json.loads(self.trace.read_text())
        with patch(PREFIX+'trace_storage.open_trace',return_value=trace), \
             patch(PREFIX+'trace_storage.release_trace_cache') as release, \
             patch(PREFIX+'audit.audit_motion',return_value=self.audit) as audit:
            result=execute_motion_audit(self.prepared,self.trace,self.output)
        self.assertEqual(result,self.audit)
        self.assertEqual(audit.call_args.args[0],prepared)
        self.assertIs(audit.call_args.args[1],trace['samples'])
        self.assertEqual(len(audit.call_args.args[1]),3)
        release.assert_called_once_with(trace)


class ConstructionWorkflowAuditRoutingTests(unittest.TestCase):
    def fixture(self):
        # Reuse the established cached-physics fixture; no solver or geometry
        # process runs while testing which audit path the workflow selects.
        from .test_slim_workflow import SlimWorkflowTests
        fixture=SlimWorkflowTests()
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        workflow,old,geometry,request,trace=fixture._partial_physics_fixture()
        return fixture,workflow

    def test_construction_routes_only_motion_audit_to_bounded_child(self):
        from ..slim_harness.construction_budget import initialize_budget
        fixture,workflow=self.fixture()
        workflow.construction=initialize_budget(workflow.project)
        workflow.building=True
        def audit_child(prepared_path,trace_path,output_path,**kwargs):
            audit={'passed':True,'violations':[]}
            write_json_atomic(output_path,audit)
            return audit
        with patch(PREFIX+'construction_audit.run_motion_audit_bounded',side_effect=audit_child) as bounded:
            result,native,auditor=fixture._exercise_partial_fixture(workflow)
        self.assertEqual(result['status'],'validation_only')
        native.assert_not_called()
        self.assertEqual(auditor.call_count,1)  # The unchanged initial-pose audit.
        bounded.assert_called_once()
        self.assertEqual(bounded.call_args.args[0].name,'prepared-scene.json')
        self.assertEqual(bounded.call_args.args[1].name,'physical.trace.json')
        self.assertEqual(bounded.call_args.kwargs['policy']['simulation_timeout_seconds'],180)
        self.assertEqual(bounded.call_args.kwargs['policy']['audit_timeout_seconds'],600)

    def test_evaluation_keeps_direct_initial_and_motion_audits(self):
        fixture,workflow=self.fixture()
        with patch(PREFIX+'construction_audit.run_motion_audit_bounded') as bounded:
            result,native,auditor=fixture._exercise_partial_fixture(workflow)
        self.assertEqual(result['status'],'validation_only')
        native.assert_not_called()
        self.assertEqual(auditor.call_count,2)
        bounded.assert_not_called()

    def test_light_closeout_ledger_keeps_saved_motion_audit(self):
        from ..slim_harness.construction_budget import initialize_budget
        fixture,workflow=self.fixture()
        workflow.construction=initialize_budget(workflow.project)
        workflow.building=False
        with patch(PREFIX+'construction_audit.run_motion_audit_bounded') as bounded:
            result,native,auditor=fixture._exercise_partial_fixture(workflow)
        self.assertEqual(result['status'],'validation_only')
        native.assert_not_called()
        self.assertEqual(auditor.call_count,2)
        bounded.assert_not_called()


if __name__=='__main__':unittest.main()
