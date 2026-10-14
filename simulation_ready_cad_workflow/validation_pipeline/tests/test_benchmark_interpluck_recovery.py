"""两次真实针接触之间的恢复观测规则，不测试或改写机械动力学。"""

import unittest
from benchmark_templates.observations import InterPluckRecoveryObserver


class InterPluckRecoveryTests(unittest.TestCase):
    def test_first_pluck_is_not_recovery_claim(self):
        observer = InterPluckRecoveryObserver({"tine": 0.003})
        observer.sample({"tine": 0.004}, {"tine"})
        event = observer.pluck("tine")
        self.assertFalse(event["restore_applicable"])
        self.assertEqual(event["restore_before_m"], 0.0)
        self.assertEqual(event["pre_contact_displacement_m"], 0.003)

    def test_contact_zero_is_excluded_and_free_interval_minimum_used(self):
        observer = InterPluckRecoveryObserver({"tine": 0.0})
        observer.sample({"tine": 0.004}, {"tine"})
        observer.pluck("tine")
        observer.sample({"tine": 0.0}, {"tine"})
        observer.sample({"tine": 0.002}, set())
        observer.sample({"tine": 0.0001}, set())
        observer.sample({"tine": 0.001}, set())
        observer.sample({"tine": 0.004}, {"tine"})
        event = observer.pluck("tine")
        self.assertTrue(event["restore_applicable"])
        self.assertTrue(event["restore_interval_observed"])
        self.assertEqual(event["restore_before_m"], 0.0001)
        self.assertEqual(event["pre_contact_displacement_m"], 0.001)

    def test_no_free_interval_preserves_actual_value_and_gap(self):
        observer = InterPluckRecoveryObserver({"tine": 0.002})
        observer.sample({"tine": 0.004}, {"tine"})
        observer.pluck("tine")
        observer.sample({"tine": 0.005}, {"tine"})
        event = observer.pluck("tine")
        self.assertFalse(event["restore_interval_observed"])
        self.assertEqual(event["restore_before_m"], 0.004)

    def test_each_tine_and_each_interval_is_independent(self):
        observer = InterPluckRecoveryObserver({"a": 0.0, "b": 0.002})
        observer.sample({"a": 0.003, "b": 0.002}, {"a"})
        observer.pluck("a")
        observer.sample({"a": 0.0001, "b": 0.004}, {"b"})
        self.assertFalse(observer.pluck("b")["restore_applicable"])
        observer.sample({"a": 0.006, "b": 0.004}, {"a", "b"})
        self.assertEqual(observer.pluck("a")["restore_before_m"], 0.0001)
        observer.sample({"a": 0.003, "b": 0.004}, set())
        observer.sample({"a": 0.006, "b": 0.004}, {"a"})
        self.assertEqual(observer.pluck("a")["restore_before_m"], 0.003)


class ContactMotionTests(unittest.TestCase):
    def event(self, pin="p", tine="a"):
        return dict(
            contact_key=("A", pin, tine, 0),
            pin_id=pin,
            tine_id=tine,
            tine_note_id=0,
            contacts=[{"normal_force_n": 0.1}],
        )

    def test_fixed_contact_never_becomes_pluck(self):
        from benchmark_templates.observations import ContactMotionObserver

        m = ContactMotionObserver({"a": (0.0, 0.0, 0.0)}, 1e-6)
        for t in (0.1, 0.2, 0.3):
            accepted, expired = m.sample(t, [self.event()], {"a": (0.0, 0.0, 0.0)})
            self.assertEqual(accepted, [])
        accepted, expired = m.sample(0.4, [], {"a": (0.0, 0.0, 0.0)})
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0]["contact_start_s"], 0.1)

    def test_contact_then_actual_motion_confirms_original_identity(self):
        from benchmark_templates.observations import ContactMotionObserver

        m = ContactMotionObserver({"a": (0.0, 0.0, 0.0)}, 1e-6)
        self.assertEqual(m.sample(0.1, [self.event()], {"a": (0.0, 0.0, 0.0)})[0], [])
        accepted, _ = m.sample(0.2, [self.event()], {"a": (0.0, 2e-6, 0.0)})
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["contact_start_s"], 0.1)
        self.assertEqual(accepted[0]["motion_confirmed_s"], 0.2)
        self.assertEqual(accepted[0]["pin_id"], "p")
        self.assertEqual(m.sample(0.3, [self.event()], {"a": (0.0, 3e-6, 0.0)})[0], [])

    def test_same_phase_chord_requires_each_real_tine_to_move(self):
        from benchmark_templates.observations import (
            ContactMotionObserver,
            CyclicPluckObserver,
        )

        m = ContactMotionObserver({"a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0)}, 1e-6)
        events = [self.event("p", "a"), self.event("q", "b")]
        accepted, _ = m.sample(
            0.1, events, {"a": (0.0, 2e-6, 0.0), "b": (0.0, 2e-6, 0.0)}
        )
        self.assertEqual(len(accepted), 2)
        o = CyclicPluckObserver(
            {
                "A": [
                    dict(pin_id="p", note_id=0, phase_rad=0.1),
                    dict(pin_id="q", note_id=0, phase_rad=0.1),
                ]
            }
        )
        o.select("A", 0.0)
        for e in accepted:
            e["restore_before_m"] = 0.0
        self.assertEqual(len(o.observe("A", 0.1, accepted)), 2)
        self.assertFalse(o.order_resolution_gap)

    def test_motion_is_vector_displacement_not_radius_change(self):
        from benchmark_templates.observations import ContactMotionObserver

        m = ContactMotionObserver({"a": (0.001, 0.0, 0.0)}, 1e-6)
        accepted, _ = m.sample(0.1, [self.event()], {"a": (0.0, 0.001, 0.0)})
        self.assertEqual(len(accepted), 1)
        self.assertGreater(accepted[0]["max_contact_motion_m"], 0.001)


if __name__ == "__main__":
    unittest.main()
