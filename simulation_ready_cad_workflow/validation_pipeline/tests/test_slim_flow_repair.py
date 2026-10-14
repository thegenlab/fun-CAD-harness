"""Regressions from the four engineering trials; synthetic records are fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import trimesh
from PIL import Image

from ..slim_harness.contracts import DeliveryError
from ..slim_harness.review import review_submission
from ..slim_harness.review_phase import next_evidence_round
from ..slim_harness.trial_cache import input_key,store_trial,reuse_trial
from ..slim_harness.implementation import native_implementation_binding
from ..slim_harness.numerical import NUMERICAL_POLICY
from ..slim_harness.trace_storage import TraceWriter,open_trace
from ..slim_harness.evidence import render_views
from ..mechanism.io import write_json_atomic


class FlowRepairTests(unittest.TestCase):
    def fixture(self):
        from .test_slim_workflow import SlimWorkflowTests
        helper=SlimWorkflowTests("test_missing_or_partial_scene_is_delivery_feedback")
        helper.setUp();self.addCleanup(helper.doCleanups)
        return helper

    def test_compile_error_returns_to_author_without_review(self):
        f=self.fixture();workflow,old,geometry,request,trace=f._partial_physics_fixture()
        prefix="simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow."
        with patch(prefix+"compile_scene",side_effect=DeliveryError("handle_pivot region_a is not near real source material")):
            result,native,auditor=f._exercise_partial_fixture(workflow)
        self.assertEqual("NEEDS_REVISION",result["status"])
        self.assertIn("handle_pivot",result["issues"][0])
        workflow._review.assert_not_called();native.assert_not_called()





    def test_development_cache_skips_physics_but_not_formal_audit(self):
        f=self.fixture();workflow,old,geometry,request,trace=f._partial_physics_fixture()
        draft=f.workspace/".runtime_runs/author_tools/simulate-fixture/tests/move/physical.trace.json"
        draft.parent.mkdir(parents=True)
        shutil.move(old/"tests/move/physical.trace.json",draft)
        write_json_atomic(draft.parent/"request.json",{**request,"output":str(draft)})
        (old/"tests/move/request.json").unlink()
        store_trial(request["scene"],request["test"],draft,workflow.chrono_python)
        result,native,auditor=f._exercise_partial_fixture(workflow)
        self.assertEqual("validation_only",result["status"])
        native.assert_not_called()
        self.assertEqual(2,auditor.call_count)
        self.assertEqual(trace["samples"],auditor.call_args.args[1])

    def test_old_preflight_rejection_does_not_replace_current_code_gates(self):
        f=self.fixture();workflow,old,geometry,request,trace=f._partial_physics_fixture()
        write_json_atomic(old/"preflight_review.json",{"status":"needs_revision"})
        workflow._review=Mock(return_value={"status":"needs_revision","observations":[],"test_coverage":[]})
        result,native,auditor=f._exercise_partial_fixture(workflow)
        self.assertEqual("validation_only",result["status"])
        workflow._review.assert_not_called()
        self.assertEqual(2,auditor.call_count)
        native.assert_not_called()


class TrialReuseTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[3]/".runtime_runs/flow_repair_tests";root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/"source.step";self.source.write_text("source fixture")
        self.mesh=self.root/"source.obj";self.mesh.write_text("mesh fixture")
        self.test={"id":"motion","duration_s":.02,"inputs":[],"loads":[]}
        self.prepared={"units":"m","bodies":[{"id":"body","source_file":str(self.source),"mesh_file":str(self.mesh),
            "source_stat":[1,2,3],"mass_kg":1.,"position_m":[0,0,0]}],"tests":[self.test]}
        self.trace=self.root/".runtime_runs/author_tools/simulate-fixture/tests/motion/physical.trace.json"
        self.trace.parent.mkdir(parents=True)
        write_json_atomic(self.trace.parent/"request.json",{"scene":self.prepared,"test":self.test,"output":str(self.trace)})
        with TraceWriter(self.trace,2) as writer:
            for t in [0,.02]:writer.append({"time_s":t,"body_poses":{}},{"time_s":t})
            writer.commit({"test_id":"motion","physical_pass":True,"failure":None,
                           "numerical_policy":NUMERICAL_POLICY,
                           "numerical_settings":{"requested_max_step_s":.001,"effective_initial_step_s":.001},
                           "numerical_attempts":[{"max_step_s":.001}],
                           "execution_implementation":native_implementation_binding(),"elapsed_seconds":100.})
        store_trial(self.prepared,self.test,self.trace,"python")

    def test_identical_files_at_new_paths_reuse_physics(self):
        new=copy.deepcopy(self.prepared)
        copied=self.root/"copied.step";shutil.copy2(self.source,copied)
        new["bodies"][0]["source_file"]=str(copied);new["bodies"][0]["source_stat"]=[4,5,6]
        target=self.root/"formal.trace.json"
        result=reuse_trial(self.root,new,self.test,target,"python")
        self.assertIsNotNone(result)
        self.assertEqual(2,len(result["samples"]))
        self.assertEqual(100.,result["physics_cache_reuse"]["original_physics_elapsed_seconds"])

    def test_coarse_development_trace_does_not_override_finer_policy(self):
        target=self.root/"fine.trace.json"
        self.assertIsNone(reuse_trial(self.root,self.prepared,self.test,target,"python",max_step_s=.0005))
        self.assertFalse(target.exists())

    def test_matching_fine_development_trace_can_be_reused(self):
        request=json.loads((self.trace.parent/"request.json").read_text())
        request["numerical_settings"]={"max_step_s":.0005}
        write_json_atomic(self.trace.parent/"request.json",request)
        trace=json.loads(self.trace.read_text())
        trace["numerical_settings"]={"requested_max_step_s":.0005,"effective_initial_step_s":.0005}
        trace["numerical_attempts"]=[{"max_step_s":.0005}]
        write_json_atomic(self.trace,trace)
        self.assertIsNotNone(store_trial(self.prepared,self.test,self.trace,"python"))
        self.assertIsNotNone(reuse_trial(self.root,self.prepared,self.test,self.root/"fine.trace.json","python",max_step_s=.0005))

    def test_changed_force_mass_or_scope_cannot_reuse(self):
        for field in ["force","mass","scope"]:
            prepared=copy.deepcopy(self.prepared);test=copy.deepcopy(self.test)
            if field=="force":test["loads"]=[{"force_n":[1,0,0]}]
            if field=="mass":prepared["bodies"][0]["mass_kg"]=2
            if field=="scope":prepared["validation_scope"]={"signature":"different"}
            with self.subTest(field=field):
                self.assertIsNone(reuse_trial(self.root,prepared,test,self.root/(field+".json"),"python"))

    def test_changed_source_or_trace_bytes_cannot_reuse(self):
        self.source.write_text("changed source")
        self.assertIsNone(reuse_trial(self.root,self.prepared,self.test,self.root/"changed.json","python"))
        self.source.write_text("source fixture")
        data=json.loads(self.trace.read_text());data["elapsed_seconds"]=0;write_json_atomic(self.trace,data)
        self.assertIsNone(reuse_trial(self.root,self.prepared,self.test,self.root/"tampered.json","python"))

    def test_other_runtime_or_implementation_cannot_reuse(self):
        self.assertIsNone(reuse_trial(self.root,self.prepared,self.test,self.root/"other.json","other-python"))
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.trial_cache.native_implementation_binding",return_value={"changed":"engine"}):
            self.assertIsNone(reuse_trial(self.root,self.prepared,self.test,self.root/"impl.json","python"))


class RenderingDepthTests(unittest.TestCase):
    def test_front_pointer_is_not_hidden_by_a_different_part_of_frame(self):
        root=Path(__file__).resolve().parents[3]/".runtime_runs/flow_render_tests";root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            directory=Path(directory)
            plate=trimesh.creation.box(extents=[.02,.001,.02])
            protrusion=trimesh.creation.box(extents=[.002,.002,.002]);protrusion.apply_translation([.018,-.03,0])
            frame=trimesh.util.concatenate([plate,protrusion])
            pointer=trimesh.creation.box(extents=[.003,.001,.012]);pointer.apply_translation([0,-.004,0])
            bodies=[]
            for name,mesh,color in [("frame",frame,[.1,.2,.3]),("pointer",pointer,[.9,.6,.1])]:
                path=directory/(name+".obj");mesh.export(path)
                bodies.append({"id":name,"mesh_file":str(path),"position_m":[0,0,0],"rotation_wxyz":[1,0,0,0],"color":color})
            output=render_views({"bodies":bodies},directory/"view.png")
            pixels=np.array(Image.open(output).convert("RGB"))
            front=pixels[:len(pixels)//2,pixels.shape[1]//2:]
            gold=(front[:,:,0]>120)&(front[:,:,0]>front[:,:,2]*1.5)&(front[:,:,1]>front[:,:,2])
            self.assertGreater(int(gold.sum()),50,"The near gold pointer should remain visible in the front view")


if __name__=="__main__":unittest.main()
