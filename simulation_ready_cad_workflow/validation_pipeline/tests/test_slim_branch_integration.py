"""Cross-branch contracts: frozen/generated tasks, real source holes, retry meaning."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from ..slim_harness.cli import start, initialize_project
from ..slim_harness.contracts import load_scene, DeliveryError
from ..slim_harness.frozen_experiment import load_experiment, bind_scene
from ..slim_harness.source_queries import contains_material
from ..slim_harness.source_support import check_supports
from ..slim_harness.numerical import run_fixed_attempts
from ..slim_harness.trace_storage import open_trace
from ..slim_harness.trace_storage import file_digest
from ..slim_harness.publication import read_result


class BranchIntegrationTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[3]/".runtime_runs"/"branch_integration_tests"
        root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def experiment(self):
        return {"schema_version":"slim_frozen_experiment_v1","task_text":"Move a block.","units":"m",
                "criteria":[{"id":"move","kind":"function","statement":"Move a block","source_quote":"Move a block",
                             "verification":"Measure actual displacement"}],"reviewer_summary":"Researcher test specification",
                "tests":[{"id":"move","purpose":"Displacement under load","duration_s":.01,
                          "assertions":[{"id":"moved","purpose":"Displacement","criterion_ids":["move"],
                                         "expression":"last(signal('block.position.x_m')) > 0.01"}]}]}

    def test_frozen_delivery_without_tests_uses_researcher_tests(self):
        source=self.root/"source";source.mkdir()
        (source/"part.obj").write_text("source bytes")
        scene={"schema_version":"slim_scene_v1","units":"m","bodies":[{"id":"block","source":"part.obj"}]}
        path=source/"scene.json";path.write_text(json.dumps(scene))
        experiment=self.root/"experiment.json";experiment.write_text(json.dumps(self.experiment()))
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli.initialize_project"):
            start(self.root,"frozen","Move a block.",scene_path=path,experiment_path=experiment)
        project=self.root/".runtime_runs/projects/frozen"
        frozen=load_experiment(project)
        delivered=load_scene(project/"runs/r01/design/scene.json",experiment=frozen)
        self.assertEqual(self.experiment()["tests"],delivered["tests"])
        self.assertEqual(scene,json.loads(path.read_text()),"Supplied source scene must remain unchanged")
        for field,value in [("tests",[]),("units","mm"),("gravity_m_s2",[0,0,0])]:
            changed={**delivered,field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):bind_scene(changed,frozen)
        saved=project/"frozen_experiment.json";saved.write_text(saved.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"changed"):load_experiment(project)

    def test_generated_mode_keeps_proposed_tests(self):
        scene={"tests":[{"id":"author_test"}]}
        self.assertEqual(scene,bind_scene(scene,None))

    def test_frozen_initialization_skips_generation_but_uses_main_workflow(self):
        source=self.root/"source";source.mkdir()
        (source/"part.obj").write_text("geometry fixture, not executed")
        scene={"schema_version":"slim_scene_v1","units":"m","bodies":[{"id":"block","source":"part.obj"}]}
        path=source/"scene.json";path.write_text(json.dumps(scene))
        experiment=self.root/"experiment.json";experiment.write_text(json.dumps(self.experiment()))
        prefix="simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli."
        with patch(prefix+"initialize_project"):
            start(self.root,"frozen","Move a block.",scene_path=path,experiment_path=experiment)
        project=self.root/".runtime_runs/projects/frozen"
        with patch(prefix+"initial_task_criteria") as generate,patch(prefix+"SlimWorkflow") as workflow:
            workflow.return_value.run.return_value={"status":"EVIDENCE_REQUIRED"}
            result=initialize_project(self.root,project)
        generate.assert_not_called()
        workflow.return_value.run.assert_called_once_with(one_revision=True)
        self.assertEqual("EVIDENCE_REQUIRED",result["status"])
        self.assertEqual(self.experiment()["criteria"],json.loads((project/"criteria.json").read_text())["criteria"])
        # Publication fixture only: review and physical outcomes are synthetic;
        # exercise file eligibility, not mechanical certification.
        run=project/"runs/r01"
        bindings={str(p):file_digest(p) for p in [run/"design/scene.json",run/"design/part.obj",project/"frozen_experiment.json"]}
        result={"schema_version":"slim_harness_result_v1","revision":1,"status":"PASS",
                "source_bindings":bindings,"criteria_sha256":file_digest(project/"criteria.json"),
                "tests":[{"physical_pass":True,"geometry_audit":{"passed":True}}],"review":{}}
        (run/"result.json").write_text(json.dumps(result))
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.review.review_passes",return_value=True):
            self.assertTrue(read_result(run)["eligible_pass"])
            frozen=project/"frozen_experiment.json"
            original=frozen.read_bytes();frozen.write_bytes(original+b" ")
            self.assertEqual("STALE_EVIDENCE",read_result(run)["status"])
            frozen.write_bytes(original)
            changed=json.loads((run/"design/scene.json").read_text())
            changed["tests"][0]["duration_s"]=.02
            (run/"design/scene.json").write_text(json.dumps(changed))
            result["source_bindings"][str(run/"design/scene.json")]=file_digest(run/"design/scene.json")
            (run/"result.json").write_text(json.dumps(result))
            self.assertEqual("STALE_EVIDENCE",read_result(run)["status"])



    def test_cavity_winding_is_not_repaired_into_material(self):
        outer=trimesh.creation.box(extents=[.02]*3)
        inner=trimesh.creation.box(extents=[.01]*3);inner.invert()
        shell=trimesh.util.concatenate([outer,inner])
        self.assertEqual([False,True],contains_material(shell,[[0,0,0],[.008,0,0]]).tolist())


    def support_fixture(self,filled=False):
        shaft=trimesh.creation.cylinder(radius=.004,height=.02,sections=128)
        bore=(trimesh.creation.cylinder(radius=.008,height=.016,sections=128) if filled else
              trimesh.creation.annulus(r_min=.0041,r_max=.008,height=.016,sections=128))
        bodies=[]
        for name,mesh in [("shaft",shaft),("bore",bore)]:
            path=self.root/(name+".obj");mesh.export(path)
            bodies.append({"id":name,"mesh_file":str(path),"position_m":[0,0,0],"rotation_wxyz":[1,0,0,0]})
        support={"max_clearance":.0002}
        for name,radius,length in [("shaft",.004,.02),("bore",.0041,.016)]:
            support[name]={"body":name,"center":[0,0,0],"axis":[0,0,1],"radius":radius,"length":length}
        joint={"id":"bearing","type":"revolute","body_a":"shaft","body_b":"bore",
               "axis":[0,0,1],"anchor":[0,0,0],"support":support}
        return {"units":"m","joints":[joint]},{"bodies":bodies}

    def test_true_bearing_still_requires_axial_review(self):
        scene,geometry=self.support_fixture()
        report=check_supports(scene,geometry)[0]
        self.assertEqual("radial_support_checked",report["status"])
        self.assertEqual("independent_review_required",report["axial_retention"])

    def test_false_bore_and_wrong_joint_axis_are_rejected(self):
        scene,geometry=self.support_fixture(filled=True)
        with self.assertRaises(ValueError):check_supports(scene,geometry)

    def test_declared_lengths_cannot_bridge_a_real_axial_gap(self):
        scene,geometry=self.support_fixture()
        shaft=trimesh.creation.cylinder(radius=.004,height=.014,sections=128)
        bore=trimesh.creation.annulus(r_min=.0041,r_max=.008,height=.014,sections=128)
        shaft.export(self.root/"shaft.obj");bore.export(self.root/"bore.obj")
        geometry["bodies"][1]["position_m"]=[0,0,.0148]
        proof=scene["joints"][0]["support"]
        proof["shaft"]["length"]=.02;proof["bore"]["length"]=.02
        proof["bore"]["center"]=[0,0,.0148]
        with self.assertRaises(ValueError):check_supports(scene,geometry)
        scene,geometry=self.support_fixture()
        scene["joints"][0]["axis"]=[1,0,0]
        with self.assertRaises(ValueError):check_supports(scene,geometry)

    def test_numeric_retry_preserves_test_and_each_attempt(self):
        test={"duration_s":.01,"step_s":.0008,"loads":[{"force_n":[1,0,0]}]};original=copy.deepcopy(test)
        steps=[]
        def once(path,step):
            steps.append(step)
            trace={"samples":[],"signals":{},"physical_pass":False,"failure":{"kind":"contact_penetration"},"step_s":step}
            path.write_text(json.dumps(trace));return trace
        result=run_fixed_attempts(once,test,self.root/"physical.trace.json")
        self.assertEqual([.0008,.0004,.0002],steps)
        self.assertEqual(original,test)
        self.assertTrue(result["numerical_failure_unresolved"])
        self.assertTrue(all(Path(row["trace_file"]).is_file() for row in result["numerical_attempts"]))

    def test_function_failure_does_not_retry(self):
        def once(path,step):
            value={"samples":[],"signals":{},"physical_pass":False,"failure":None,"assertions":[{"passed":False}]}
            path.write_text(json.dumps(value));return value
        result=run_fixed_attempts(once,{"step_s":.01},self.root/"physical.trace.json")
        self.assertEqual(1,len(result["numerical_attempts"]))
        self.assertFalse(result["numerical_failure_unresolved"])
        self.assertFalse(open_trace(self.root/"physical.trace.json")["physical_pass"])


if __name__=="__main__":unittest.main()
