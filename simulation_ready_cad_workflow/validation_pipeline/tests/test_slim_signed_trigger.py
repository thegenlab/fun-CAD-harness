"""Signed-trigger numerical regression only; no CAD, native or author execution."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import unittest
import numpy as np
from . import test_slim_compact_measurements as fixtures

kernel=fixtures.kernel
C=-math.atan(.06/16.55)
OMEGA=2*math.pi/43200
DT=.0005
F3=('F3_measured_automatic_release','F3_no_early_impact_since_ARM','F3_first_impact_matches_actual_setting')


def evaluate(release=12.,first=12.25,offset=0.,extra=(),no_reference=False):
    p=fixtures.profile('automatic_alarm_and_stop');p['step_s']=DT;p['phase_initial_rad']+=offset
    data=fixtures.columns(p);t=data['time_s']
    data['trigger_slide.coordinate']=.001*(t>=release)
    if no_reference:data['escape_bearing.coordinate'][:]=0
    points={round(when/DT):[fixtures.contact()] for when in (first,*extra)}
    result=kernel.evaluate_clock_measurement('automatic_alarm_and_stop',data,
        fixtures.contacts(t,points),fixtures.contact_schema(),measurement_profile=p)
    return result,fixtures.decisions(result)


class SignedTrigger(unittest.TestCase):
    def test_remaining_overlap_rejects_release_and_impact(self):
        for phase in (C+OMEGA,0.,-C-OMEGA):
            with self.subTest(phase=phase):
                result,checks=evaluate(first=12.,offset=phase-C)
                for name in F3:self.assertFalse(checks[name],name)
                self.assertAlmostEqual(phase,result['measurements']['release_phase_rad'])

    def test_first_early_reader_candidate_is_not_skipped(self):
        result,checks=evaluate(release=11.,first=12.25)
        self.assertEqual(11.,result['measurements']['release_s'])
        self.assertFalse(checks[F3[0]])
        self.assertTrue(checks[F3[1]])
        self.assertFalse(checks[F3[2]])

    def test_only_two_solver_steps_allow_early_phase(self):
        for steps,expected in ((1.5,True),(2.5,False)):
            with self.subTest(early_steps=steps):
                result,checks=evaluate(first=12.,offset=steps*DT*OMEGA)
                for name in F3:self.assertEqual(expected,checks[name],name)
                self.assertAlmostEqual(2*DT*OMEGA,result['measurements']['numerical_phase_allowance_rad'])

    def test_four_half_beats_only_allow_later_time_and_more_negative_phase(self):
        for first,expected in ((13.9995,True),(14.01,False)):
            with self.subTest(first=first):
                result,checks=evaluate(first=first)
                self.assertTrue(checks[F3[0]]);self.assertTrue(checks[F3[1]])
                self.assertEqual(expected,checks[F3[2]])
                self.assertAlmostEqual(2.,result['measurements']['first_impact_latency_s'])
                self.assertAlmostEqual(C-2*OMEGA,result['measurements']['first_impact_minimum_phase_rad'])
        # Within the allowed time but below c-4B*omega by half a step: no
        # extra epsilon may extend the negative lower phase boundary.
        _,checks=evaluate(first=13.9995,offset=-1.5*DT*OMEGA)
        self.assertFalse(checks[F3[2]])

    def test_one_step_release_order_allowance_and_clear_release_are_both_required(self):
        _,checks=evaluate(first=12.-DT)
        self.assertTrue(checks[F3[2]])
        _,checks=evaluate(first=12.-2*DT,offset=-.5*DT*OMEGA)
        self.assertTrue(checks[F3[0]]);self.assertTrue(checks[F3[1]])
        self.assertFalse(checks[F3[2]])
        _,checks=evaluate(release=13.,first=12.25)
        self.assertTrue(checks[F3[0]]);self.assertTrue(checks[F3[1]])
        self.assertFalse(checks[F3[2]])

    def test_release_diagnostic_remains_when_continuity_reference_is_missing(self):
        result,checks=evaluate(release=11.,no_reference=True)
        self.assertFalse(checks['F5_sustained_reference'])
        self.assertFalse(checks[F3[0]])
        self.assertEqual(C,result['measurements']['release_boundary_phase_rad'])
        self.assertEqual(-C,result['measurements']['blade_half_angle_rad'])
        self.assertEqual(11.,result['measurements']['release_s'])

    def test_new_auto_profiles_require_negative_boundary_without_a_second_width(self):
        for value in (None,0.,-C,float('inf'),float('nan'),True):
            p=fixtures.profile('automatic_alarm_and_stop')
            if value is None:p.pop('release_boundary_phase_rad')
            else:p['release_boundary_phase_rad']=value
            with self.subTest(value=value),self.assertRaises(kernel.MeasurementCapabilityError):
                kernel.validate_measurement_profile(p,'automatic_alarm_and_stop')
        p=fixtures.profile('automatic_alarm_and_stop');p['blade_half_angle_rad']=-C
        with self.assertRaises(kernel.MeasurementCapabilityError):kernel.validate_measurement_profile(p,'automatic_alarm_and_stop')
        old=copy.deepcopy(kernel.LEGACY_MEASUREMENT_PROFILES['automatic_alarm_and_stop'])
        binding={'schema_version':'slim_alarm_measurement_binding_v3','test_id':'automatic_alarm_and_stop',
                 'duration_s':old['duration_s'],'step_s':old['step_s'],'measurement_profile':old}
        with self.assertRaises(kernel.MeasurementCapabilityError):kernel.measurement_profile_for_registration(binding)
        with self.assertRaises(kernel.MeasurementCapabilityError):
            kernel.evaluate_clock_measurement('automatic_alarm_and_stop',{},measurement_profile=fixtures.profile('automatic_alarm_and_stop'),legacy_v2=True)

if __name__=='__main__':unittest.main()
