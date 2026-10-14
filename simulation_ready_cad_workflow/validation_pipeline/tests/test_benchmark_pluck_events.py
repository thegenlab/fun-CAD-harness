"""和弦不能被当作单事件；原生记录器保留逐件事件供人审。以下为接口单测。"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from benchmark_templates.observations import CyclicPluckObserver
from benchmark_templates.native import NativeModel, Signal, run

PROGRAMS = {
    1: [
        {"pin_id": "a", "note_id": 0, "phase_rad": 0.0},
        {"pin_id": "b", "note_id": 2, "phase_rad": 0.0},
        {"pin_id": "c", "note_id": 1, "phase_rad": math.pi},
    ]
}


def event(pin, tine, note=None, restore=0.0):
    return {
        "pin_id": pin,
        "tine_id": tine,
        "tine_note_id": tine if note is None else note,
        "restore_before_m": restore,
    }


class PluckEventsTests(unittest.TestCase):
    def test_chord_identity_order_and_repeated_contact(self):
        observer = CyclicPluckObserver(PROGRAMS)
        observer.select(1, 0.0)
        records = observer.observe(1, 0.0, [event("a", 0), event("b", 2)])
        self.assertEqual(len(records), 2)
        self.assertEqual(observer.count, 2)
        self.assertFalse(observer.order_resolution_gap)
        self.assertEqual(observer.phase_gap, 0.0)
        self.assertEqual(observer.pin_note, observer.tine_note)
        self.assertEqual(observer.observe(1, 0.01, [event("a", 0), event("b", 2)]), [])
        observer.observe(1, math.pi, [event("c", 1)])
        self.assertEqual(observer.phase_gap, 0.0)
        observer.observe(1, math.tau, [event("a", 0), event("b", 2)])
        self.assertEqual(observer.count, 5)
        self.assertEqual(observer.repluck, 2)

    def test_wrong_note_missing_chord_member_and_restore_are_not_hidden(self):
        observer = CyclicPluckObserver(PROGRAMS)
        observer.select(1, 0.0)
        observer.observe(1, 0.0, [event("a", 3, restore=0.004)])
        self.assertEqual(abs(observer.pin_note - observer.tine_note), 3.0)
        self.assertEqual(observer.restore_error, 0.004)
        observer.observe(1, math.pi, [event("c", 1)])
        self.assertEqual(observer.phase_gap, math.pi)

    def test_cross_turn_events_use_unfolded_phase_and_report_timing_resolution(self):
        programs = {
            1: [
                {"pin_id": "late", "note_id": 0, "phase_rad": 6.2},
                {"pin_id": "early", "note_id": 1, "phase_rad": 0.1},
            ]
        }
        observer = CyclicPluckObserver(programs)
        observer.select(1, 6.0)
        records = observer.observe(
            1, math.tau + 0.11, [event("early", 1), event("late", 0)]
        )
        self.assertEqual([x["pin_id"] for x in records], ["late", "early"])
        self.assertEqual(observer.phase_gap, 0.0)
        self.assertTrue(observer.order_resolution_gap)
        resolved = CyclicPluckObserver(programs)
        resolved.select(1, 6.0)
        resolved.observe(1, 6.2, [event("late", 0)])
        resolved.observe(1, math.tau + 0.11, [event("early", 1), event("late", 0)])
        self.assertFalse(resolved.order_resolution_gap)

    def test_native_report_preserves_multiple_observation_events(self):
        class System:
            time = 0.0

            def GetChTime(self):
                return self.time

            def Update(self, *args):
                pass

            def DoStepDynamics(self, dt):
                self.time += dt
                return True

        model = NativeModel(System(), {"x": Signal(lambda: 0.0, "source_scalar")}, {})

        def observe(t, dt):
            model.observation_events.extend(
                [
                    {"time_s": t, "pin": "a", "tine": 0},
                    {"time_s": t, "pin": "b", "tine": 2},
                ]
            )

        model.after_step = observe
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = run(model, [], duration_s=0.001, step_s=0.001, output=output)
            saved = json.loads((output / "report.json").read_text())
        self.assertEqual(len(result["observation_events"]), 2)
        self.assertEqual(result["observation_events"], saved["observation_events"])
        self.assertEqual(result["events"], [])


if __name__ == "__main__":
    unittest.main()
