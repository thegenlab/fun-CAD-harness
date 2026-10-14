from __future__ import annotations

import unittest
import numpy as np

from ..slim_harness.measurements import cycles, evaluate, evaluate_assertions


class SlimCycleMeasurementsTests(unittest.TestCase):
    def test_two_complete_sinusoidal_cycles_from_either_endpoint(self):
        phase = np.linspace(0, 4*np.pi, 2001)
        low_start = .02*(1-np.cos(phase))
        self.assertEqual(2, cycles(low_start, .03, .001))
        self.assertEqual(2, cycles(.04-low_start, .03, .001))

    def test_phase_shifted_recording_does_not_count_cut_boundary_fragments(self):
        two_periods = .02*(1-np.cos(np.linspace(np.pi/2, 4*np.pi+np.pi/2, 2001)))
        three_periods = .02*(1-np.cos(np.linspace(np.pi/2, 6*np.pi+np.pi/2, 3001)))
        self.assertEqual(1, cycles(two_periods, .03, .001))
        self.assertEqual(2, cycles(three_periods, .03, .001))

    def test_half_cycles_and_partial_reversals_are_not_accumulated(self):
        self.assertEqual(0, cycles([0, .04, .02], .03, .001))
        self.assertEqual(0, cycles([0, .04, .02, .03, 0], .03, .001))
        walking_jiggles = [value for offset in np.arange(0, .1, .004) for value in (offset, offset+.008, offset+.004)]
        self.assertGreater(np.ptp(walking_jiggles), .03)
        self.assertEqual(0, cycles(walking_jiggles, .03, .001))
        self.assertEqual(0, cycles(.004*np.sin(np.linspace(0, 200*np.pi, 10001)), .03, .001))

    def test_drifted_endpoints_do_not_count_as_return(self):
        self.assertEqual(0, cycles([0, .04, .005, .045, .010, .050, .015], .03, .001))
        self.assertEqual(1, cycles([0, .04, .0005], .03, .001))
        self.assertEqual(0, cycles([0, .04, .0015], .03, .001))

    def test_roundoff_jitter_and_plateaus_do_not_create_extra_extrema(self):
        phase = np.linspace(0, 4*np.pi, 2001)
        clean = -np.cos(phase)
        jitter = np.where(np.arange(len(clean)) % 2, 2e-16, -2e-16)
        self.assertEqual(2, cycles(clean+jitter, 1.9, 0))
        self.assertEqual(2, cycles([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], 1, 0))
        self.assertEqual(0, cycles([0, 0, 0], 1, 0))

    def test_counted_cycles_do_not_share_a_leg(self):
        self.assertEqual(1, cycles([0, 1, 0, 1], 1, 0))
        self.assertEqual(2, cycles([0, 1, 0, 1, 0], 1, 0))

    def test_missing_nonfinite_or_invalid_cycle_inputs_are_not_measurements(self):
        for values, stroke, tolerance in (([], 1, 0), ([0, np.nan, 0], 1, 0),
                                           ([0, np.inf, 0], 1, 0), ([0, 1, 0], 0, 0),
                                           ([0, 1, 0], 1, -1), ([0, 1, 0], np.inf, 0)):
            with self.subTest(values=values, stroke=stroke, tolerance=tolerance):
                with self.assertRaises(ValueError):
                    cycles(values, stroke, tolerance)


class SlimWindowMeasurementsTests(unittest.TestCase):
    def setUp(self):
        self.signals = {"time_s": [0, 1, 2, 3, 4], "x": [0, 1, 2, 3, 4], "speed": [2, 2, 2, 2, 2]}

    def test_windows_are_inclusive_and_do_not_interpolate(self):
        for expression in ("first(window(signal('x'), 1, 3)) == 1",
                           "last(window(signal('x'), 1, 3)) == 3",
                           "first(window(signal('x'), 1.5, 3.5)) == 2",
                           "last(window(signal('x'), 1.5, 3.5)) == 3"):
            with self.subTest(expression=expression):
                self.assertTrue(evaluate(expression, self.signals)["passed"])

    def test_window_time_coordinates_survive_nesting_arithmetic_and_integral(self):
        for expression in ("integral(window(signal('speed'), 1, 3)) == 4",
                           "integral(abs(window(signal('speed'), 1, 3)) * 2) == 8",
                           "last(window(after(signal('x'), 1), 2, 3)) == 3",
                           "integral(before(after(signal('speed'), 1), 3)) == 4",
                           "mean(window(window(signal('x'), 1, 4), 2, 3)) == 2.5"):
            with self.subTest(expression=expression):
                self.assertTrue(evaluate(expression, self.signals)["passed"])

    def test_empty_reversed_and_nonfinite_windows_fail(self):
        for expression in ("max(window(signal('x'), 9, 10)) > 0",
                           "max(window(signal('x'), 3, 1)) > 0",
                           "max(window(signal('x'), 0, 1e309)) > 0"):
            with self.subTest(expression=expression):
                rows = evaluate_assertions({"assertions": [{"id": "window", "expression": expression}]}, self.signals)
                self.assertFalse(rows[0]["passed"])

    def test_cycle_helper_composes_with_window_and_records_the_count(self):
        times = np.linspace(0, 6, 3001)
        signals = {"time_s": times, "piston": .02*(1-np.cos(np.pi*times))}
        result = evaluate("cycles(window(signal('piston'), .5, 5.5), .03, .001) >= 2", signals)
        self.assertTrue(result["passed"])
        self.assertEqual([2., 2.], result["observed_operands"])

    def test_existing_scalar_reductions_and_masks_keep_their_behavior(self):
        for expression in ("span(signal('x')) == 4", "mean(signal('speed')) == 2",
                           "rms(signal('speed')) == 2", "min(signal('x')) == 0",
                           "max(signal('x') * (signal('time_s') <= 2)) == 2"):
            with self.subTest(expression=expression):
                self.assertTrue(evaluate(expression, self.signals)["passed"])


if __name__ == "__main__":
    unittest.main()
