"""Saved-trace sampling and unchanged-pair reuse without physics reruns."""
from __future__ import annotations

import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..slim_harness.audit import audit_motion, audit_saved_motion
from ..slim_harness.cli import main
from ..slim_harness.geometry import prepare_geometry


def pose(x):
    return {"position_abs_m": [x, 0., 0.], "rotation_abs_quat_wxyz": [1., 0., 0., 0.]}


class SavedMotionAuditTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[3]/".runtime_runs/four_gaps_pair_20260912/audit_tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def geometry(self):
        import xml.parsers.expat
        from build123d import Box, Compound, Pos, export_step
        export_step(Compound(children=[Pos(-15,0,0)*Box(10,10,10),
                                      Pos(-5,0,0)*Box(10,10,10),
                                      Pos(30,0,0)*Box(5,5,5)]), self.root/"assembly.step")
        scene = {"schema_version":"slim_scene_v1", "units":"mm", "bodies":[
            {"id":key,"source":"assembly.step","solid_ids":[f"solid_{index}"],"fixed":False}
            for index,key in enumerate(("a","b","c"),1)]}
        path = self.root/"scene.json"
        path.write_text(json.dumps(scene), encoding="utf-8")
        geometry = prepare_geometry(path, self.root/"geometry")
        self.assertEqual([], geometry["errors"])
        return geometry

    def samples(self, count=3):
        return [{"time_s":index*.1,"body_poses":{"a":pose(-.015),"b":pose(-.005),"c":pose(.030)}}
                for index in range(count)]

    def test_uniform_whole_trace_grid_retains_exact_endpoints(self):
        samples = [{"time_s":index*.01,"phase_id":str(index),"contact_count":index%2,
                    "body_poses":{"a":pose(0)}} for index in range(104)]
        observed = []
        def kernel(_prepared, selected, _output):
            observed.extend(row["time_s"] for row in selected)
            return {"passed":True,"checked_solver_samples":len(selected),"violations":[]}
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.audit.audit_motion", side_effect=kernel):
            report = audit_saved_motion({"bodies":[{"id":"a"}]},samples,self.root/"sampled.json")
        self.assertEqual([samples[index]["time_s"] for index in list(range(0,101,10))+[103]],observed)
        self.assertEqual(list(range(0,101,10))+[103],report["audited_source_indices"])
        self.assertEqual((104,12,12),(report["available_solver_samples"],report["candidate_solver_samples"],report["checked_solver_samples"]))
        self.assertFalse(report["sampling_policy"]["phase_boundaries_forced"])

    def test_invalid_unsampled_pose_or_time_still_fails_before_geometry(self):
        samples = [{"time_s":index*.01,"body_poses":{"a":pose(0)}} for index in range(21)]
        for field in ("body_poses", "time_s"):
            bad = copy.deepcopy(samples)
            bad[1].pop(field)
            with self.assertRaises(RuntimeError):
                audit_saved_motion({"bodies":[{"id":"a"}]},bad,self.root/f"bad-{field}.json")
            self.assertFalse((self.root/f"bad-{field}.json").exists())

    def test_invalid_time_sequence_is_rejected_before_geometry(self):
        for times in ([], [0., float("nan"), .2], [0., float("inf"), .2], [0., .2, .1]):
            with self.subTest(times=times):
                samples = [{"time_s":t,"body_poses":{"a":pose(0)}} for t in times]
                output = self.root/"invalid-time.json"
                with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.audit.audit_motion") as kernel:
                    with self.assertRaises(RuntimeError):
                        audit_saved_motion({"bodies":[{"id":"a"}]},samples,output)
                    kernel.assert_not_called()
                self.assertFalse(output.exists())

    def test_unchanged_dynamic_pair_reused_but_new_third_body_collision_checked(self):
        geometry, samples = self.geometry(), self.samples()
        samples[2]["body_poses"]["c"] = pose(-.005)
        report = audit_motion(geometry,samples,self.root/"third-body.json")
        self.assertFalse(report["passed"])
        self.assertTrue(any({row["body_a"],row["body_b"]} == {"b","c"} for row in report["violations"]))
        self.assertEqual(.2,report["violations"][0]["time_s"])
        self.assertEqual(2,report["boolean_checks"])
        self.assertGreaterEqual(report["unchanged_pair_checks_reused"],4)

    def test_small_actual_relative_motion_invalidates_unchanged_pair_cache(self):
        geometry, samples = self.geometry(), self.samples(2)
        samples[1]["body_poses"]["a"] = pose(-.015+1e-6)
        report = audit_motion(geometry,samples,self.root/"small-move.json")
        self.assertFalse(report["passed"])
        self.assertEqual(.1,report["violations"][0]["time_s"])
        self.assertGreater(report["max_overlap_m3"],1e-12)

    def test_public_audit_reads_saved_trace_and_rejects_stale_source(self):
        geometry = self.geometry()
        trace_path = self.root/"physical.trace.json"
        samples = self.samples(5)
        trace_path.write_text(json.dumps({"samples":samples,"signals":{"time_s":[s["time_s"] for s in samples]},"physical_pass":False}),encoding="utf-8")
        request_path = self.root/"request.json"
        request_path.write_text(json.dumps({"scene":geometry,"test":{"id":"existing","duration_s":.4},"output":str(trace_path)}),encoding="utf-8")
        output = self.root/"public-audit.json"
        with redirect_stdout(io.StringIO()) as stream:
            code = main(["audit",str(request_path),"--output",str(output)])
        self.assertEqual(0,code)
        self.assertFalse(json.loads(stream.getvalue())["certified"])
        report = json.loads(output.read_text())
        self.assertFalse(report["physics_rerun"])
        self.assertEqual(4,report["unchanged_pair_checks_reused"]//3)
        (self.root/"assembly.step").write_text("changed source",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"binding changed"):
            main(["audit",str(request_path),"--output",str(self.root/"stale.json")])


if __name__ == "__main__":
    unittest.main()
