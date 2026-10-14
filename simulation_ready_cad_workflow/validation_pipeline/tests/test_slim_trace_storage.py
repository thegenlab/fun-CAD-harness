"""Real disk round trips and existing-reader checks; no Chrono execution."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from ..slim_harness.audit import audit_motion
from ..slim_harness.diagnostics import same_recorded_poses
from ..slim_harness.evidence import export_replay
from ..slim_harness.geometry import prepare_geometry
from ..slim_harness.measurements import evaluate_assertions
from ..slim_harness.trace_storage import (
    SampleRows, TraceStorageError, TraceWriter, file_digest, open_trace, reuse_trace, trace_artifacts,
)


class SlimTraceStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, count=7):
        samples = [{"time_s": i * .01,
                    "body_poses": {"arm": {"position_abs_m": [i * .001, 2., 3.],
                                             "rotation_abs_quat_wxyz": [1., 0., 0., 0.]}},
                    "force_evidence": {"available": False, "reason": "not_sampled"},
                    "contact_step": {"step": i, "time_s": i * .01, "contacts": []},
                    "fea_nodes": {"source_node": [i * .0001, 0., 0.]}}
                   for i in range(count)]
        signals = {"time_s": [i * .01 for i in range(count)], "speed": [2.] * count,
                   "pin.active": [float(i < count // 2) for i in range(count)]}
        metadata = {"test_id": "data_fixture", "step_s": .01, "requested_duration_s": (count - 1) * .01,
                    "physical_pass": False, "assertions": [{"id": "native_false", "passed": False}],
                    "measurement_contact_schema": {"sampling": "every_solver_step", "fixture_only": True},
                    "force_evidence_sampling": {"solver_indices": [0, count - 1]}, "connections": [{"id": "pin"}]}
        return samples, signals, metadata

    def write(self, name="trace.json", *, count=7, expected=None):
        samples, signals, metadata = self.fixture(count)
        path = self.root / name
        with TraceWriter(path, expected or count, chunk_rows=2) as writer:
            for index, sample in enumerate(samples):
                row = {key: values[index] for key, values in signals.items()}
                if index % 2:
                    row = dict(reversed(list(row.items())))
                writer.append(sample, row)
            writer.commit(metadata)
        return path, samples, signals, metadata

    def test_all_samples_signals_and_opaque_contact_fea_fields_round_trip(self):
        path, samples, signals, metadata = self.write()
        view = open_trace(path)
        self.assertEqual(samples, list(view["samples"]))
        self.assertEqual(samples[1:], list(view["samples"][1:]))
        self.assertEqual(samples[::-1], list(view["samples"][::-1]))
        self.assertEqual(samples[-1], view["samples"][-1])
        self.assertEqual(metadata["measurement_contact_schema"], view["measurement_contact_schema"])
        self.assertFalse(view["physical_pass"])
        self.assertEqual(7, view["storage"]["recorded_samples"])
        for name, values in signals.items():
            np.testing.assert_array_equal(values, view["signals"][name])
        manifest = json.loads(path.read_text())
        self.assertNotIn("samples", manifest)
        self.assertNotIn("signals", manifest)

    def test_existing_measurements_keep_identical_results_on_disk_columns(self):
        path, _, signals, _ = self.write()
        test = {"assertions": [{"id": "true", "expression": "integral(window(signal('speed'), .01, .05)) == .08"},
                               {"id": "false", "expression": "max(signal('speed')) < 1"}]}
        expected = evaluate_assertions(test, signals)
        self.assertEqual(expected, evaluate_assertions(test, open_trace(path)["signals"]))
        self.assertFalse(expected[1]["passed"])

    def test_shortened_run_exposes_only_actual_rows_and_compacts_each_column(self):
        path, samples, signals, _ = self.write(count=3, expected=100)
        view = open_trace(path)
        self.assertEqual(100, view["storage"]["expected_samples"])
        self.assertEqual(3, len(view["samples"]))
        self.assertEqual(samples, list(view["samples"]))
        self.assertEqual(3 * len(signals) * 8, view["storage"]["files"]["signals"]["bytes"])
        for name, values in signals.items():
            np.testing.assert_array_equal(values, view["signals"][name])

    def test_missing_or_modified_side_data_is_not_accepted(self):
        path, _, _, _ = self.write()
        metadata = json.loads(path.read_text())
        signal_path = path.parent / metadata["storage"]["files"]["signals"]["path"]
        with signal_path.open("r+b") as stream:
            stream.write(b"corrupt!")
        with self.assertRaisesRegex(TraceStorageError, "binding mismatch"):
            open_trace(path)
        signal_path.unlink()
        with self.assertRaises(FileNotFoundError):
            open_trace(path)

    def test_legacy_inline_trace_is_readable_and_reuse_preserves_data_bindings(self):
        path, samples, signals, metadata = self.write()
        cached = self.root / "reused" / "physical.trace.json"
        view = reuse_trace(path, cached)
        self.assertEqual(samples, list(open_trace(cached)["samples"]))
        self.assertEqual(set(trace_artifacts(path)[1:]), set(trace_artifacts(cached)[1:]))
        self.assertTrue(same_recorded_poses(open_trace(path), view))
        legacy = {**metadata, "schema_version": "slim_physical_trace_v1", "samples": samples, "signals": signals}
        old_path = self.root / "old.json"
        old_path.write_text(json.dumps(legacy), encoding="utf-8")
        self.assertEqual(legacy, open_trace(old_path))
        copied = self.root / "old-copy.json"
        self.assertEqual(samples, reuse_trace(old_path, copied)["samples"])

    def test_regular_exception_preserves_data_prefix_without_publishing_pass(self):
        path = self.root / "interrupted.json"
        samples, signals, _ = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
            with TraceWriter(path, 7, chunk_rows=4) as writer:
                writer.append(samples[0], {name: values[0] for name, values in signals.items()})
                raise RuntimeError("fixture interruption")
        self.assertFalse(path.exists())
        data = json.loads(path.with_suffix(".data").joinpath("data.json").read_text())
        self.assertEqual(1, data["storage"]["recorded_samples"])
        self.assertNotIn("physical_pass", data)

    def test_missing_channel_or_mismatched_time_is_not_filled_with_zeros(self):
        for name, sample, row in (("missing", {"time_s": 1.}, {"time_s": 1.}),
                                  ("time", {"time_s": 2.}, {"time_s": 1., "x": 3.})):
            with self.subTest(name=name):
                with self.assertRaises(TraceStorageError):
                    with TraceWriter(self.root / (name + ".json"), 3) as writer:
                        writer.append({"time_s": 0.}, {"time_s": 0., "x": 2.})
                        writer.append(sample, row)
                self.assertFalse((self.root / (name + ".json")).exists())

    def test_replay_can_select_frames_while_audit_still_sees_the_middle_step(self):
        first = trimesh.creation.box(extents=[.01, .01, .01])
        second = first.copy()
        second.apply_translation([.1, 0, 0])
        first.export(self.root / "fixed.obj")
        second.export(self.root / "moving.obj")
        scene = {"schema_version": "slim_scene_v1", "units": "m", "bodies": [
            {"id": "fixed", "source": "fixed.obj", "fixed": True},
            {"id": "moving", "source": "moving.obj", "fixed": False}]}
        scene_path = self.root / "scene.json"
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        geometry = prepare_geometry(scene_path, self.root / "geometry")
        self.assertEqual([], geometry["errors"])
        path = self.root / "all-steps.json"
        with TraceWriter(path, 5, chunk_rows=2) as writer:
            for i, x in enumerate((.1, .1, 0., .1, .1)):
                poses = {"fixed": {"position_abs_m": [0, 0, 0], "rotation_abs_quat_wxyz": [1, 0, 0, 0]},
                         "moving": {"position_abs_m": [x, 0, 0], "rotation_abs_quat_wxyz": [1, 0, 0, 0]}}
                writer.append({"time_s": i * .01, "body_poses": poses}, {"time_s": i * .01, "pin.active": float(i < 3)})
            writer.commit({"test_id": "middle-overlap", "physical_pass": False,
                           "force_evidence_sampling": {"solver_indices": [0, 4]}, "connections": [{"id": "pin"}]})
        view = open_trace(path)
        replay = export_replay(geometry, view, self.root / "replay")
        frames = json.loads(Path(replay["replay_file"]).read_text())["branches"]["physical"]["frames"]
        self.assertEqual([0, 4], [frame["source_sample_index"] for frame in frames])
        self.assertFalse(frames[-1]["joint_active"]["pin"])
        audit = audit_motion(geometry, view["samples"], self.root / "audit.json")
        self.assertFalse(audit["passed"])
        self.assertEqual(5, audit["available_solver_samples"])
        self.assertEqual(.02, audit["violations"][0]["time_s"])


if __name__ == "__main__":
    unittest.main()
