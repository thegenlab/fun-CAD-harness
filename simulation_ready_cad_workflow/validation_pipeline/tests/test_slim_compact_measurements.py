"""Portable numerical/interface tests. No CAD/native execution or D PASS evidence."""
from pathlib import Path
import copy
import importlib
import json
import math
import sys
import tempfile
import types
import unittest
import numpy as np

from simulation_ready_cad_workflow.validation_pipeline.slim_harness import alarm_measurements as kernel
from simulation_ready_cad_workflow.validation_pipeline.slim_harness import measurement_registry as registry

def profile(test_id='independent_alarm_setting'):
    value = copy.deepcopy(kernel.LEGACY_MEASUREMENT_PROFILES[test_id])
    value.update(duration_s=20. if test_id == 'independent_alarm_setting' else 30., step_s=.001,
                 reference_window_s=[2.,8.], minimum_reference_intervals=8)
    for name in ('escape','center'):
        value['channels'][name]['scale'] = -1
    if test_id == 'independent_alarm_setting':
        value['operation_windows_s'] = {'alarm_setting':[10.,11.]}
    else:
        value.update(operation_windows_s={'OFF':[22.,23.]}, reader_reference_window_s=[2.,8.],
                     release_search_window_s=[9.,21.], armed_window_s=[8.,22.],
                     ringing_window_s=[13.,19.], before_off_window_s=[20.,21.8],
                     after_off_window_s=[25.,29.], phase_initial_rad=-math.atan(.06/16.55)+12*2*math.pi/43200,
                     release_boundary_phase_rad=-math.atan(.06/16.55))
    return value


def source_test(test_id='independent_alarm_setting'):
    p = profile(test_id)
    inputs = ([{'joint':'set_alarm_time','mode':'speed','value':-.1,'start_s':10.,'end_s':11.}]
              if test_id == 'independent_alarm_setting' else [
                  {'joint':'alarm_switch','mode':'position','value':.006,'start_s':8.,'end_s':8.5},
                  {'joint':'alarm_switch','mode':'position','value':0,'start_s':22.,'end_s':23.}])
    return {'id':test_id, 'duration_s':p['duration_s'], 'step_s':p['step_s'], 'inputs':inputs,
            'assertions':[{'id':'native_fixture','expression':'1 == 1','criterion_ids':['F5']}], 'loads':[]}


def columns(p, frame_rate=0.):
    t = np.arange(round(p['duration_s']/p['step_s'])+1)*p['step_s']
    values = {'time_s':t, 'frame.angle.z_rad':frame_rate*t}
    for name, old_column, increment in kernel.CHANNELS:
        spec = p['channels'][name]
        # Fixture models the actual A=frame,B=wheel native sign, not a supplied result.
        relative = 2*increment*t if name in ('escape','center') else -2*increment*t
        values[old_column] = relative + (frame_rate*t if name in ('minute','hour') else 0.)
        body = {'escape':'escape_arbor','center':'center_arbor','minute':'cannon_minute','hour':'hour_sleeve'}[name]
        values[body+'.angle.z_rad'] = -2*increment*t + frame_rate*t
    values['trigger_slide.coordinate'] = .001*(t >= 12.)
    values['alarm_setting_ring.angle.z_rad'] = np.zeros_like(t)
    return values


def contact():
    return {'body_a':'alarm_hammer','body_b':'left_bell',
            'normal_impulse_n_s':.001,'closing_speed_before_impulse_m_s':.1}


def contacts(t, points=None, skip=None):
    for i in range(1,len(t)):
        if i != skip:
            yield {'step':i,'time_s':float(t[i]),'contacts':(points or {}).get(i,[])}


def contact_schema():
    return {**kernel.CONTACT_SCHEMA, 'covered_body_pairs':[
        ['alarm_hammer','left_bell'],['alarm_hammer','right_bell']],
        'calibration_signature':'numerical-fixture-only'}


def impacts(p):
    points = {}
    for i, time in enumerate([12.25,12.75,13.,14.,15.,16.,20.25,20.75,21.25,21.75]):
        row = contact(); row['body_b'] = kernel.BELLS[i%2]
        points[round(time/p['step_s'])] = [row]
    return points


def decisions(result):
    return {row['id']:row['passed'] for row in result['assertions']}


class CompactMeasurements(unittest.TestCase):
    def test_native_joint_sign_mapping_and_common_frame_motion(self):
        p = profile(); data = columns(p,frame_rate=.01)
        result = kernel.evaluate_clock_measurement('independent_alarm_setting',data,measurement_profile=p)
        self.assertTrue(all(decisions(result).values()))
        self.assertIsNone(result['task_verdict'])
        for name in ('escape','center'):
            bad = copy.deepcopy(p);bad['channels'][name]['scale'] = 1
            answer = decisions(kernel.evaluate_clock_measurement('independent_alarm_setting',data,measurement_profile=bad))
            self.assertFalse(all(answer.values()),name)
        for name, body in [('escape','escape_arbor'),('center','center_arbor')]:
            p['channels'][name] = {'signal':body+'.angle.z_rad','relative_to':'frame.angle.z_rad','scale':1}
        self.assertTrue(all(decisions(kernel.evaluate_clock_measurement('independent_alarm_setting',data,measurement_profile=p)).values()))

    def test_each_channel_stall_across_both_operation_edges_fails(self):
        for test_id, operation, edges in [('independent_alarm_setting','alarm_setting',(9.5,11.)),
                                          ('automatic_alarm_and_stop','OFF',(21.5,23.))]:
            p = profile(test_id)
            for name, column, increment in kernel.CHANNELS:
                for start in edges:
                    with self.subTest(test=test_id,channel=name,edge=start):
                        data = columns(p); t = data['time_s']; sign = 1 if name in ('escape','center') else -1
                        data[column] = sign*2*increment*(t-np.clip(t-start,0,1.))
                        time,dt,sig = kernel._signals(data,p)
                        answer,_ = kernel._continuity(time,dt,sig,p['operation_windows_s'],p)
                        self.assertFalse(answer[f'F5_{name}_maximum_gap_at_{operation}'])

    def test_reversal_missing_bracket_and_unresolved_sampling_stay_failures(self):
        t = np.arange(6)*.1
        np.testing.assert_array_equal(kernel.advance_times(t,np.array([0,-1.1,-.8,-1.15,-1.8,-2.1]),1),[.1,.5])
        with self.assertRaises(kernel.MeasurementCapabilityError):
            kernel.advance_times(np.array([0,.1]),np.array([0,-2.1]),1)
        self.assertTrue(math.isinf(kernel.bracketing_max_gap(np.array([0,.5,1]),.5,2)))
        self.assertEqual(3.,kernel.bracketing_max_gap(np.array([0,.5,3.5,4.]),1,4))

    def test_short_alarm_uses_bound_phase_real_episodes_and_off_windows(self):
        p = profile('automatic_alarm_and_stop'); data = columns(p); points = impacts(p)
        def evaluate():
            return kernel.evaluate_clock_measurement('automatic_alarm_and_stop',data,
                contacts(data['time_s'],points),contact_schema(),measurement_profile=p)
        answer = evaluate()
        self.assertEqual(14,len(answer['assertions']));self.assertTrue(all(decisions(answer).values()))
        p['phase_initial_rad'] += .1
        self.assertFalse(decisions(evaluate())['F3_first_impact_matches_actual_setting'])
        p['phase_initial_rad'] -= .1
        points[26000] = [contact()]
        self.assertFalse(decisions(evaluate())['F4_no_actual_impacts_after_OFF'])

    def test_support_first_positive_classification_and_missing_records_unchanged(self):
        t = np.arange(201)*.001
        resting = contact();resting['closing_speed_before_impulse_m_s'] = 0.
        points = {100:[resting],101:[contact()],102:[contact()],104:[contact()]}
        self.assertEqual([104],[s[0] for s in kernel._strikes(t,contacts(t,points),contact_schema())])
        with self.assertRaises(kernel.MeasurementCapabilityError):
            kernel._strikes(t,contacts(t,points,skip=50),contact_schema())
        p = profile(); data = {k:v[::2] for k,v in columns(p).items()}
        with self.assertRaises(kernel.MeasurementCapabilityError):
            kernel.evaluate_clock_measurement('independent_alarm_setting',data,measurement_profile=p)

    def test_invalid_windows_sources_scales_and_unmatched_operation_rejected(self):
        mutations = [lambda p:p.update(reference_window_s=[2,21]),
                     lambda p:p.update(minimum_reference_intervals=0),
                     lambda p:p['channels']['center'].update(signal='alarm_hammer.angle.z_rad'),
                     lambda p:p['channels']['center'].update(scale=2),
                     lambda p:p['channels']['center'].update(signal=['center_bearing.coordinate']),
                     lambda p:p.update(reference_window_s=[2.,2.0001]),
                     lambda p:p.update(reference_window_s=[10.,14.]),
                     lambda p:p['operation_windows_s'].update(alarm_setting=[12,13]),
                     lambda p:p.update(step_s=.002),
                     lambda p:p.update(expression='arbitrary_source_code()')]
        for mutation in mutations:
            p=profile();mutation(p)
            with self.subTest(profile=p),self.assertRaises(kernel.MeasurementCapabilityError):
                kernel.validate_measurement_profile(p,'independent_alarm_setting',source_test())

    def test_missing_position_value_is_zero_not_arm(self):
        ident='automatic_alarm_and_stop';p=profile(ident);test=source_test(ident)
        del test['inputs'][0]['value']
        with self.assertRaisesRegex(kernel.MeasurementCapabilityError,'ARM window'):
            kernel.validate_measurement_profile(p,ident,test)
        test=source_test(ident);del test['inputs'][1]['value']
        kernel.validate_measurement_profile(p,ident,test)


class ExplicitRegistration(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.project=Path(tmp.name)
        self.design=self.project/'runs/r14/design';self.design.mkdir(parents=True)
        self.test=source_test();self.p=profile()
        self.files={self.project/'task.md':'frozen task',self.project/'criteria.json':'{}',
                    self.project/'measurement-policy.json':'{"duration_policy":"fixture-only"}',
                    self.design/'plan.json':'{"reviewed_numerical_plan":"fixture-only"}',
                    self.design/'requirements.json':'{"continuity":"all four actual channels"}'}
        for path,text in self.files.items():path.write_text(text,encoding='utf-8')
        self.scene={'units':'mm','tests':[self.test]};self.write(self.design/'scene.json',self.scene)
        self.binding={'schema_version':'slim_alarm_measurement_binding_v3','id':'explicit-short-fixture',
            'implementation':'slim_harness.alarm_measurements','test_id':self.test['id'],
            'duration_s':20.,'step_s':.001,'criterion_ids':['F5'],'measurement_profile':self.p,
            'impact_event_rule':'first_positive_impulse_classifies_episode',
            'implementation_sha256':kernel.implementation_sha256(),
            'criteria_sha256':registry.digest(self.project/'criteria.json'),'task_sha256':registry.digest(self.project/'task.md'),
            'source_plan':{'path':'plan.json'},'source_requirements':{'path':'requirements.json'},
            'source_plan_bindings':{name:registry.digest(self.design/name) for name in ('plan.json','requirements.json')},
            'measurement_policy_bindings':{'measurement-policy.json':registry.digest(self.project/'measurement-policy.json')},
            'source_test_spec_sha256':registry.value_digest(self.test),
            'test_spec_sha256':registry.value_digest(self.test)}
        self.seal()
        self.registration={'schema_version':'slim_project_measurement_registration_v1',
            'criteria_sha256':self.binding['criteria_sha256'],'task_sha256':self.binding['task_sha256'],
            'bindings':[self.binding]}

    def write(self,path,value):path.write_text(json.dumps(value),encoding='utf-8')
    def seal(self):self.binding['signature']=kernel.registration_signature(self.binding)
    def install_fixture(self):self.write(self.project/registry.REGISTRATION_FILE,self.registration)

    def test_profile_test_source_plan_and_policy_are_all_bound(self):
        self.install_fixture()
        self.assertEqual({self.test['id']:self.binding},registry.load_registered_measurements(self.project,self.design/'scene.json'))
        for path in self.files:
            data=path.read_bytes();path.write_bytes(data+b' changed')
            with self.subTest(path=path),self.assertRaises(kernel.MeasurementCapabilityError):
                registry.load_registered_measurements(self.project,self.design/'scene.json')
            path.write_bytes(data)
        self.binding['measurement_profile']['channels']['center']['scale']=1
        with self.assertRaises(kernel.MeasurementCapabilityError):registry.validate_runtime_registration(self.binding,self.test)

    def test_no_new_profile_inference_or_policy_omission(self):
        for field in ('measurement_profile','measurement_policy_bindings','source_plan','source_requirements'):
            original=self.binding.pop(field);self.seal()
            with self.subTest(field=field),self.assertRaises(kernel.MeasurementCapabilityError):
                registry.validate_runtime_registration(self.binding,self.test)
            self.binding[field]=original;self.seal()
        self.binding['measurement_profile']['duration_s']=21;self.seal()
        with self.assertRaises(kernel.MeasurementCapabilityError):registry.validate_runtime_registration(self.binding,self.test)

    def test_measurements_keep_native_rejection_and_cache_binding(self):
        data=columns(self.p)
        for failure in (None,{'kind':'actual_native_failure'}):
            trace={'test_id':self.test['id'],'failure':failure,'physical_pass':False,
                   'measurement_registration_signature':self.binding['signature'],
                   'assertions':[{'id':'native','passed':False}], 'signals':data,
                   'samples':[{'time_s':0.}]*len(data['time_s'])}
            trace.update(registry.evaluate_native_trace(trace,self.binding))
            self.assertFalse(trace['physical_pass']);registry.validate_trace_measurements(trace,self.binding)
            self.assertEqual({},registry.evaluate_native_trace(trace,self.binding))
            trace['physical_pass']=True
            with self.assertRaises(kernel.MeasurementCapabilityError):registry.validate_trace_measurements(trace,self.binding)

    def test_original_v2_registration_uses_only_the_explicit_historical_grid(self):
        old=copy.deepcopy(self.binding);old.update(schema_version='slim_alarm_measurement_binding_v2',duration_s=75.)
        old.pop('measurement_profile');old.pop('measurement_policy_bindings')
        p=kernel.measurement_profile_for_registration(old)
        self.assertEqual([30.,60.],p['reference_window_s']);self.assertEqual(40,p['minimum_reference_intervals'])
        self.assertEqual(1,p['channels']['center']['scale'])
        old['duration_s']=20.
        with self.assertRaises(kernel.MeasurementCapabilityError):kernel.measurement_profile_for_registration(old)


if __name__ == '__main__':unittest.main()
