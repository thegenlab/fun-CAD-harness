from pathlib import Path
import copy
import json
import tempfile
import unittest

import trimesh

from ..slim_harness.evidence import export_replay
from ..slim_harness.diagnostics import same_recorded_poses


class SlimForceEvidenceTests(unittest.TestCase):
    def fixture(self,root):
        mesh=root/"original.obj"
        trimesh.creation.box(extents=[.01,.02,.03]).export(mesh)
        body={"id":"arm","mesh_file":str(mesh),"fixed":False,"source_solid_ids":["solid_1"],
              "position_m":[1,2,3],"rotation_wxyz":[1,0,0,0],"color":[.5,.5,.5,1]}
        region={"points":[[.001,0,0],[.002,0,0]],"radius":.004}
        trace={"samples":[{"time_s":i*.01,"body_poses":{"arm":{"position_abs_m":[1+i*.001,2,3],"rotation_abs_quat_wxyz":[1,0,0,0]}}} for i in range(5)],
               "connections":[{"id":"pin","type":"revolute","body_a":"arm","body_b":"ground","region_a":region,"region_b":region}],
               "signals":{"pin.active":[1,1,1,0,0]}}
        return {"bodies":[body]},trace

    def test_force_frames_keep_their_native_solver_indices_and_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);geometry,trace=self.fixture(root)
            trace["force_evidence_sampling"]={"solver_indices":[0,2,4]}
            trace["samples"][0]["force_evidence"]={"available":False,"reason":"unsolved"}
            force={"available":True,"joint_reactions":[{"joint_id":"pin","active":True,
                   "force_a_abs_n":[3,4,0],"force_b_abs_n":[-3,-4,0],"torque_a_abs_nm":[0,0,2],"torque_b_abs_nm":[0,0,-2]}],
                   "contacts":[],"external_loads":[{"body_id":"arm","point_abs_m":[1,2,3],"force_abs_n":[0,0,-10],"torque_abs_nm":[0,0,0]}]}
            trace["samples"][2]["force_evidence"]=force
            trace["samples"][4]["force_evidence"]={"available":True,"joint_reactions":[],"contacts":[],"external_loads":[]}
            assets=export_replay(geometry,trace,root/"replay")
            replay=json.loads(Path(assets["replay_file"]).read_text())
            branch=replay["branches"]["physical"]
            self.assertEqual([0,2,4],[frame["source_sample_index"] for frame in branch["frames"]])
            self.assertEqual(force,branch["frames"][1]["force_evidence"])
            self.assertFalse(branch["frames"][2]["joint_active"]["pin"])
            self.assertEqual(trace["connections"],replay["connections"])
            self.assertEqual(10,branch["maxima"]["force_n"])
            self.assertEqual(2,branch["maxima"]["torque_nm"])
            self.assertTrue(branch["field_availability"]["joint_reactions"])

    def test_old_motion_does_not_invent_force_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);geometry,trace=self.fixture(root)
            assets=export_replay(geometry,trace,root/"replay")
            branch=json.loads(Path(assets["replay_file"]).read_text())["branches"]["physical"]
            self.assertFalse(branch["field_availability"]["joint_reactions"])
            self.assertTrue(branch["field_availability"]["connection_regions"])
            self.assertTrue(all("force_evidence" not in frame for frame in branch["frames"]))

    def test_geometry_audit_reuse_requires_every_pose_and_time_to_match_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            _,old=self.fixture(Path(directory));new=copy.deepcopy(old)
            new["samples"][2]["force_evidence"]={"available":True,"joint_reactions":[]}
            self.assertTrue(same_recorded_poses(old,new))
            new["samples"][2]["body_poses"]["arm"]["position_abs_m"][0]+=1e-12
            self.assertFalse(same_recorded_poses(old,new))
            new=copy.deepcopy(old);new["samples"][2]["time_s"]+=1e-12
            self.assertFalse(same_recorded_poses(old,new))
            self.assertFalse(same_recorded_poses({"samples":[]},{"samples":[]}))


if __name__=="__main__":unittest.main()
