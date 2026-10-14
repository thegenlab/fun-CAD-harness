from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from ..slim_harness import alarm_measurements as kernel
from ..slim_harness import measurement_registry as registry
from ..slim_harness.contracts import compile_scene
from ..mechanism.io import write_json_atomic
from . import test_slim_workflow as workflow_fixtures


class MeasurementRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.design = self.root/'runs/r01/design'
        self.design.mkdir(parents=True)
        (self.root/'criteria.json').write_text('{"frozen":true}')
        (self.root/'task.md').write_text('Frozen fixture task')
        (self.design/'plan.py').write_text('# Plan input only; never execute')
        self.test = {'id':'independent_alarm_setting','duration_s':75.,'step_s':.001,
                     'loads':[{'body':'frame','point':[1.,2.,3.]}]}
        self.scene = {'units':'mm','tests':[self.test]}
        write_json_atomic(self.design/'scene.json',self.scene)
        self.binding = {'schema_version':'slim_alarm_measurement_binding_v2','id':'explicit-fixture',
            'implementation':'slim_harness.alarm_measurements','test_id':self.test['id'],
            'duration_s':75.,'step_s':.001,'criterion_ids':['F5'],
            'impact_event_rule':'first_positive_impulse_classifies_episode',
            'implementation_sha256':kernel.implementation_sha256(),
            'criteria_sha256':registry.digest(self.root/'criteria.json'),
            'task_sha256':registry.digest(self.root/'task.md'),
            'source_plan_bindings':{'plan.py':registry.digest(self.design/'plan.py')},
            'source_test_spec_sha256':registry.value_digest(self.test),
            'test_spec_sha256':registry.value_digest(registry.compiled_test_spec(self.test,'mm'))}
        self.binding['signature']=kernel.registration_signature(self.binding)
        self.registration={'schema_version':'slim_project_measurement_registration_v1',
            'criteria_sha256':self.binding['criteria_sha256'],'task_sha256':self.binding['task_sha256'],
            'bindings':[self.binding]}

    def register(self):
        write_json_atomic(self.root/registry.REGISTRATION_FILE,self.registration)

    def trace(self, native_pass=True):
        t=np.arange(75001)*.001
        signals={'time_s':t,'frame.angle.z_rad':np.zeros_like(t)}
        for _name,column,increment in kernel.CHANNELS:
            signals[column]=-2*increment*t
        return {'test_id':self.test['id'],'failure':None,'physical_pass':native_pass,
            'measurement_registration_signature':self.binding['signature'],
            'assertions':[{'id':'native','passed':native_pass}],
            'signals':signals,'samples':[{'time_s':0.}]*len(t)}

    def test_scene_ids_alone_never_register_author_measurements(self):
        self.assertEqual({},registry.load_registered_measurements(self.root,self.design/'scene.json'))
        self.register()
        self.assertEqual({self.test['id']:self.binding},registry.load_registered_measurements(self.root,self.design/'scene.json'))

    def test_exact_source_plan_task_and_test_changes_reject_binding(self):
        self.register()
        for path in [self.design/'plan.py',self.root/'task.md',self.root/'criteria.json',self.design/'scene.json']:
            original=path.read_bytes()
            with self.subTest(path=path.name):
                if path.name=='scene.json':
                    changed=copy.deepcopy(self.scene);changed['tests'][0]['duration_s']=74.
                    write_json_atomic(path,changed)
                else:path.write_bytes(original+b' changed')
                with self.assertRaises(registry.MeasurementCapabilityError):
                    registry.load_registered_measurements(self.root,self.design/'scene.json')
                path.write_bytes(original)

    def test_compiled_unit_conversion_is_bound_before_physics(self):
        compiled=compile_scene(self.scene,{'bodies':[]})['tests'][0]
        registry.validate_runtime_registration(self.binding,compiled)
        self.assertEqual([.001,.002,.003],compiled['loads'][0]['point'])
        with self.assertRaises(registry.MeasurementCapabilityError):
            registry.validate_runtime_registration(self.binding,self.test)

    def test_evaluate_once_preserves_native_rejection_and_cache_does_not_append_again(self):
        for native_pass in [True,False]:
            with self.subTest(native_pass=native_pass):
                trace=self.trace(native_pass)
                trace.update(registry.evaluate_native_trace(trace,self.binding))
                self.assertIs(trace['physical_pass'],native_pass)
                self.assertEqual(5,len(trace['assertions']))
                self.assertEqual({},registry.evaluate_native_trace(trace,self.binding))
                registry.validate_trace_measurements(trace,self.binding)

    def test_cache_cannot_omit_supplement_or_change_signature_or_promote_native_false(self):
        original=self.trace(False);original.update(registry.evaluate_native_trace(original,self.binding))
        for change in [lambda t:t.update(assertions=t['native_assertions']),
                       lambda t:t.update(measurement_registration_signature='old'),
                       lambda t:t.update(physical_pass=True),
                       lambda t:t.update(measurement_error='missing contact rows')]:
            trace=dict(original);change(trace)
            with self.assertRaises(registry.MeasurementCapabilityError):
                registry.validate_trace_measurements(trace,self.binding)

    def test_actual_native_failure_remains_rejection_without_missing_contact_inference(self):
        trace=self.trace(False);trace['failure']={'kind':'contact_penetration'}
        trace.update(registry.evaluate_native_trace(trace,self.binding))
        registry.validate_trace_measurements(trace,self.binding)
        self.assertEqual('not_evaluated_after_native_failure',trace['registered_measurements']['status'])
        self.assertIs(trace['physical_pass'],False)


class MeasurementWorkflowAdoptionTests(unittest.TestCase):
    def fixture(self):
        fixture=workflow_fixtures.SlimWorkflowTests()
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        return fixture,fixture._partial_physics_fixture()

    def test_recorded_measurement_capability_failure_is_not_an_author_design_rejection(self):
        fixture,(workflow,old,_geometry,_request,trace)=self.fixture()
        trace.update(physical_pass=False,measurement_error='Actual contact records unavailable')
        write_json_atomic(old/'tests/move/physical.trace.json',trace)
        result,native,auditor=fixture._exercise_partial_fixture(workflow)
        self.assertEqual('HARNESS_ERROR',result['status'])
        native.assert_not_called();auditor.assert_called_once()
        self.assertEqual('initial-geometry-audit.json',auditor.call_args.args[2].name)

    def test_validated_bundle_cannot_bypass_new_registration(self):
        fixture,(workflow,old,_geometry,_request,_trace)=self.fixture()
        result,_,_=fixture._exercise_partial_fixture(workflow)
        self.assertEqual('validation_only',result['status'])
        write_json_atomic(fixture.run/'result.json',{'status':'EVIDENCE_REQUIRED'})
        binding={'signature':'new-required-measurement','source_plan_bindings':{'part.step':registry.digest(fixture.source)}}
        write_json_atomic(fixture.project/registry.REGISTRATION_FILE,{'fixture':'mocked explicit registration'})
        workflow._finish_validation.reset_mock()
        with patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.load_registered_measurements',return_value={'move':binding}):
            result,native,_=fixture._exercise_partial_fixture(workflow)
        self.assertEqual('HARNESS_ERROR',result['status'])
        native.assert_called_once();workflow._finish_validation.assert_not_called()

    def test_exact_raw_request_cannot_reuse_trace_without_registered_measurement(self):
        fixture,(workflow,old,_geometry,request,_trace)=self.fixture()
        binding={'signature':'new-required-measurement','source_plan_bindings':{'part.step':registry.digest(fixture.source)}}
        request['scene']['registered_measurements']={'move':binding}
        write_json_atomic(old/'tests/move/request.json',request)
        write_json_atomic(fixture.project/registry.REGISTRATION_FILE,{'fixture':'mocked explicit registration'})
        with patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.load_registered_measurements',return_value={'move':binding}):
            result,native,_=fixture._exercise_partial_fixture(workflow)
        self.assertEqual('HARNESS_ERROR',result['status'])
        native.assert_called_once();workflow._finish_validation.assert_not_called()
