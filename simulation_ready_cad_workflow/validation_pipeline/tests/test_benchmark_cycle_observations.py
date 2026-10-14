"""跨设计启动滞后、逆转和真实整周观察；不算机械物理通过。"""

import unittest, math
from benchmark_templates.observations import CycleReturnObserver


class CycleObservationTests(unittest.TestCase):
    def observer(self):
        state = [0.0, 0.0]
        o = CycleReturnObserver(lambda: state[0], lambda: state[1], 2 * math.pi)
        return state, o

    def test_no_cycle_does_not_count_as_return(self):
        s, o = self.observer()
        s[:] = [1.0, 0.5]
        o.update()
        self.assertEqual(0, o.completed)

    def test_delayed_input_returns_at_real_phase_not_time(self):
        s, o = self.observer()
        for phase, out in [
            (0.0, 0.0),
            (0.2, 0.1),
            (math.pi, 0.5),
            (2 * math.pi, 0.0),
            (3 * math.pi, 0.5),
            (4 * math.pi, 0.0),
        ]:
            s[:] = [phase, out]
            o.update()
        self.assertEqual(2, o.completed)
        self.assertEqual(0.0, o.max_return_error)

    def test_one_bad_return_cannot_be_hidden_by_later_good_cycle(self):
        s, o = self.observer()
        for phase, out in [(2 * math.pi, 0.2), (4 * math.pi, 0.0)]:
            s[:] = [phase, out]
            o.update()
        self.assertEqual(2, o.completed)
        self.assertAlmostEqual(0.2, o.max_return_error)

    def test_reverse_then_recross_does_not_duplicate_cycle(self):
        s, o = self.observer()
        for phase in (2 * math.pi, math.pi, 2 * math.pi, 4 * math.pi):
            s[:] = [phase, 0.0]
            o.update()
        self.assertEqual(2, o.completed)

    def test_crossing_between_real_samples_interpolates_without_extrapolation(self):
        s, o = self.observer()
        s[:] = [2 * math.pi - 0.1, 0.1]
        o.update()
        s[:] = [2 * math.pi + 0.1, -0.1]
        o.update()
        self.assertEqual(1, o.completed)
        self.assertAlmostEqual(0.0, o.max_return_error, places=12)

    def test_nonfinite_actual_sample_is_rejected(self):
        s, o = self.observer()
        s[0] = float("nan")
        with self.assertRaises(ValueError):
            o.update()


class PairedPhaseTests(unittest.TestCase):
    def observer(self, differences):
        from benchmark_templates.observations import PairedPeakPhaseObserver
        from benchmark_templates.native import Signal

        zero = Signal(lambda: 0.0, "source_scalar")
        o = PairedPeakPhaseObserver(zero, {"a": zero, "b": zero}, {"a": 0.036, "b": 0.036})
        o.peaks = {
            "a": [{"crank_angle_rad": i * 2 * math.pi} for i in range(len(differences))],
            "b": [{"crank_angle_rad": i * 2 * math.pi + d} for i, d in enumerate(differences)],
        }
        return o

    def test_opposite_phase_errors_cannot_cancel(self):
        o = self.observer([math.pi - 0.1, math.pi + 0.1, math.pi])
        self.assertAlmostEqual(math.pi, o.phase())
        self.assertGreater(o.max_error(math.pi), 0.03)

    def test_wrap_boundary_uses_shortest_circular_distance(self):
        o = self.observer([-0.01, -0.01, -0.01])
        self.assertLess(o.max_error(0), 0.03)


if __name__ == "__main__":
    unittest.main()
