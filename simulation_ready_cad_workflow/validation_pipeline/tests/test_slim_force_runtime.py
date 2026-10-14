"""Native, closed-OBJ calibration of force telemetry; no benchmark case runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import trimesh

from ..slim_harness.trace_storage import open_trace


CHRONO_PYTHON=Path(os.environ.get("SIMREADY_TEST_CHRONO_PYTHON",str(Path.home()/".conda/envs/chrono_py312_projectchrono/python.exe")))


@unittest.skipUnless(CHRONO_PYTHON.is_file(),"The verified native Chrono environment is unavailable")
class SlimNativeForceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=Path(__file__).resolve().parents[3]
        root=cls.workspace/".runtime_runs"/"slim_harness_20260907"/"force_runtime_tests"
        root.mkdir(parents=True,exist_ok=True)
        cls.temporary=tempfile.TemporaryDirectory(dir=root)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root=Path(cls.temporary.name)
        for name,size in (("body",[.04,.02,.06]),("ground",[.02,.02,.02]),("floor",[1,.02,1])):
            trimesh.creation.box(extents=size).export(cls.root/f"{name}.obj")

        # A smaller box rests on a roof while its front bottom corners lie
        # 60 um behind the roof body's front plane. The bodies only touch.
        trimesh.creation.box(extents=[.1,.05,.01]).export(cls.root/"roof_base.obj")
        trimesh.creation.box(extents=[.01,.01,.01]).export(cls.root/"roof_loaded.obj")
        # Same box surface, with legitimate extra vertices along its edges.
        trimesh.creation.box(extents=[.04,.02,.06]).subdivide().export(cls.root/"subdivided_body.obj")

        polygon=np.asarray([[-.01,-.01],[.01,-.01],[.01,0],[0,0],[0,.01],[-.01,.01]])
        l_vertices=np.asarray([[x,y,z] for y in (-.005,.005) for x,z in polygon])
        cap=np.asarray([[0,1,3],[1,2,3],[0,3,5],[3,4,5]])
        l_faces=cap.tolist()+(cap[:,::-1]+6).tolist()
        for i in range(6):
            j=(i+1)%6
            l_faces.extend([[i,j+6,j],[i,i+6,j+6]])
        l_mesh=trimesh.Trimesh(vertices=l_vertices,faces=l_faces,process=False)
        if not (l_mesh.is_watertight and l_mesh.is_winding_consistent and l_mesh.volume>0):
            raise AssertionError("L corner fixture must be one closed positive source solid")
        l_center=l_mesh.center_mass.copy()
        l_inertia=l_mesh.moment_inertia*(2/l_mesh.volume)
        l_mesh.apply_translation(-l_center)
        l_mesh.export(cls.root/"l_corner.obj")
        l_slider_size=np.asarray([.01006,.02,.01])
        trimesh.creation.box(extents=l_slider_size).export(cls.root/"l_slider.obj")

        def body(identifier,position,*,fixed=False,mesh="body",mass=2):
            return {"id":identifier,"position_m":position,"rotation_wxyz":[1,0,0,0],"fixed":fixed,"source_solid_ids":["solid_1"],
                    "mass_kg":mass,"inertia_kg_m2":np.diag([.0006666666667,.0008666666667,.0003333333333]).tolist(),
                    "mesh_file":str(cls.root/f"{mesh}.obj")}
        def scene(a,b,joint=None):
            return {"bodies":[a,b],"gravity_m_s2":[0,-9.81,0],"joints":[] if joint is None else [joint]}
        def joint(kind,a,b,anchor,axis,radius=.4):
            return {"id":"connection","type":kind,"body_a":a["id"],"body_b":b["id"],"anchor":anchor,"axis":axis,
                    "region_a":{"points":[(np.asarray(anchor)-a["position_m"]).tolist()]*2,"radius":radius},
                    "region_b":{"points":[(np.asarray(anchor)-b["position_m"]).tolist()]*2,"radius":radius}}
        def test(identifier,duration,**extra):
            return {"id":identifier,"duration_s":duration,"step_s":.001,
                    "assertions":[{"id":"duration","purpose":"Native calibration duration",
                                   "expression":f"last(signal('time_s')) >= {duration}"}],**extra}
        cases=[]
        a=body("loaded",[.2,.3,.4]);b=body("ground",[0,0,0],fixed=True,mesh="ground")
        link=joint("fixed",a,b,[.1,.2,.3],(np.ones(3)/np.sqrt(3)).tolist())
        cases.append({"name":"fixed","scene":scene(a,b,link),"test":test("fixed",.8)})
        a=body("loaded",[0,0,0]);b=body("ground",[0,-.2,0],fixed=True,mesh="ground")
        link=joint("revolute",a,b,[0,0,0],[0,0,1])
        cases.append({"name":"rotating","scene":scene(a,b,link),"test":test("rotating",.4,
            inputs=[{"joint":"connection","mode":"effort","value":.002,"max_effort":.002}])})
        a=body("loaded",[0,0,0]);b=body("ground",[-.2,0,0],fixed=True,mesh="ground")
        link=joint("prismatic",a,b,[0,0,0],[1,0,0],radius=.001)
        cases.append({"name":"released","scene":scene(a,b,link),"test":test("released",.04,
            inputs=[{"joint":"connection","mode":"effort","value":100,"max_effort":100}])})
        a=body("loaded",[0,.0101,0]);b=body("floor",[0,-.01,0],fixed=True,mesh="floor")
        for reverse in (False,True):
            prepared=scene(a,b)
            if reverse:prepared["bodies"].reverse()
            cases.append({"name":f"contact_{int(reverse)}","scene":prepared,"test":test(f"contact_{int(reverse)}",.3,
                loads=[{"body":"loaded","force_n":[1,0,0]}])})
        for reverse in (False,True):
            a=body("loaded",[0,.01,0]);b=body("floor",[0,-.01,0],fixed=True,mesh="floor")
            prepared=scene(a,b)
            if reverse:prepared["bodies"].reverse()
            cases.append({"name":f"touching_{int(reverse)}","scene":prepared,
                          "test":test(f"touching_{int(reverse)}",.05)})
        # Detection regression for a runtime-reachable overshot state. This
        # is deliberately not a legal initial-geometry acceptance fixture.
        a=body("loaded",[0,.01-.00006,0]);b=body("floor",[0,-.01,0],fixed=True,mesh="floor")
        cases.append({"name":"overshot_60um","scene":scene(a,b),
                      "test":test("overshot_60um",.003)})
        a=body("loaded",[0,-.02494+.005,.005],mesh="roof_loaded")
        a["inertia_kg_m2"]=(np.eye(3)*(2*.01**2/6)).tolist()
        b=body("floor",[0,0,-.005],fixed=True,mesh="roof_base")
        prepared=scene(a,b);prepared["gravity_m_s2"]=[0,0,-9.81]
        cases.append({"name":"coplanar_roof_front_corner","scene":prepared,
                      "test":test("coplanar_roof_front_corner",.01)})
        # Illegal initial states deliberately exercise runtime penetration
        # detection. Equal footprints put corners on the other box's sides;
        # boundary classification must not suppress their real 60 um overlap.
        for mesh in ("body","subdivided_body"):
            a=body("loaded",[0,.01-.00006,0],mesh=mesh)
            b=body("floor",[0,-.01,0],fixed=True,mesh="body")
            name="shared_footprint_"+mesh
            cases.append({"name":name,"scene":scene(a,b),"test":test(name,.003)})
        # The L's roof touches the box bottom while its vertical wall is truly
        # penetrated. Correcting an unrelated Z contact must not hide this.
        a=body("l_boundary",l_center.tolist(),fixed=True,mesh="l_corner")
        a["inertia_kg_m2"]=l_inertia.tolist()
        b=body("slider",[(.01-.00006)/2,0,.005],mesh="l_slider")
        b["inertia_kg_m2"]=np.diag(2/12*np.asarray([
            l_slider_size[1]**2+l_slider_size[2]**2,
            l_slider_size[0]**2+l_slider_size[2]**2,
            l_slider_size[0]**2+l_slider_size[1]**2])).tolist()
        prepared=scene(a,b);prepared["gravity_m_s2"]=[0,0,0]
        cases.append({"name":"l_corner_side_penetration","scene":prepared,
                      "test":test("l_corner_side_penetration",.003)})
        for case in cases:case["output"]=str(cls.root/(case["name"]+".trace.json"))
        request=cls.root/"cases.json";request.write_text(json.dumps(cases),encoding="utf-8")
        script="""import json,sys
from pathlib import Path
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime import simulate
for case in json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')):
    simulate(case['scene'],case['test'],Path(case['output']))
"""
        completed=subprocess.run([str(CHRONO_PYTHON),"-X","utf8","-c",script,str(request)],cwd=cls.workspace,
                                 capture_output=True,text=True,encoding="utf-8",timeout=30,
                                 creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if completed.returncode:
            raise AssertionError("Native force calibration failed: "+completed.stderr)
        cls.traces={case["name"]:open_trace(Path(case["output"])) for case in cases}
        cls.scenes={case["name"]:case["scene"] for case in cases}

    def test_fixed_joint_world_force_and_moment_balance_at_each_side(self):
        trace=self.traces["fixed"]
        self.assertIsNone(trace["failure"])
        row=trace["samples"][-1]["force_evidence"]["joint_reactions"][0]
        self.assertEqual(("loaded","ground"),(row["body_a"],row["body_b"]))
        self.assertTrue(row["active"])
        np.testing.assert_allclose(row["force_a_abs_n"],[0,19.62,0],atol=1e-5)
        np.testing.assert_allclose(row["force_b_abs_n"],[0,-19.62,0],atol=1e-5)
        np.testing.assert_allclose(row["torque_a_abs_nm"],[-1.962,0,1.962],atol=1e-5)
        np.testing.assert_allclose(np.asarray(row["force_a_abs_n"])+row["force_b_abs_n"],0,atol=1e-8)
        moment=(np.asarray(row["torque_a_abs_nm"])+row["torque_b_abs_nm"]
                +np.cross(row["point_a_abs_m"],row["force_a_abs_n"])+np.cross(row["point_b_abs_m"],row["force_b_abs_n"]))
        np.testing.assert_allclose(moment,0,atol=1e-7)

    def test_rotating_frame_uses_each_joint_frame_and_keeps_drive_separate(self):
        trace=self.traces["rotating"]
        self.assertIsNone(trace["failure"])
        frame=trace["samples"][-1]
        self.assertGreater(abs(frame["body_poses"]["loaded"]["rotation_abs_quat_wxyz"][3]),.1)
        forces=frame["force_evidence"]
        reaction=forces["joint_reactions"][0]
        np.testing.assert_allclose(reaction["force_a_abs_n"],[0,19.62,0],atol=1e-5)
        np.testing.assert_allclose(reaction["force_b_abs_n"],[0,-19.62,0],atol=1e-5)
        self.assertAlmostEqual(0,reaction["torque_a_abs_nm"][2],places=7)
        drive=next(row for row in forces["external_loads"] if row["kind"]=="drive" and row["body_id"]=="loaded")
        np.testing.assert_allclose(drive["torque_abs_nm"],[0,0,.002],atol=1e-12)
        np.testing.assert_allclose(drive["force_abs_n"],0,atol=1e-12)

    def test_released_constraint_reports_zero_instead_of_cached_old_reaction(self):
        trace=self.traces["released"]
        self.assertIsNone(trace["failure"])
        frames=[frame for frame in trace["samples"] if frame["force_evidence"]["available"]]
        reactions=[frame["force_evidence"]["joint_reactions"][0] for frame in frames]
        self.assertTrue(any(row["active"] and row["force_a_abs_n"][1]>19 for row in reactions))
        released=[row for row in reactions if not row["active"]]
        self.assertTrue(released)
        for row in released:
            for key in ("force_a_abs_n","force_b_abs_n","torque_a_abs_nm","torque_b_abs_nm"):
                self.assertEqual([0.,0.,0.],row[key])
        self.assertLess(trace["signals"]["loaded.linear_speed.y_m_s"][-1],-.1)
        for index,frame in enumerate(trace["samples"][1:],start=1):
            gravity=next(row for row in frame["force_evidence"]["external_loads"] if row["kind"]=="gravity")
            np.testing.assert_allclose(gravity["point_abs_m"],trace["samples"][index-1]["body_poses"]["loaded"]["position_abs_m"],atol=1e-12)

    def test_contact_frame_force_is_on_body_b_and_reverses_with_body_order(self):
        for name in ("contact_0","contact_1"):
            with self.subTest(name=name):
                trace=self.traces[name]
                self.assertIsNone(trace["failure"])
                frame=trace["samples"][-1]
                rows=frame["force_evidence"]["contacts"]
                self.assertTrue(rows)
                self.assertEqual(len(rows),int(trace["signals"]["contact.count"][-1]))
                loaded=np.zeros(3)
                for row in rows:
                    np.testing.assert_allclose(np.asarray(row["force_a_abs_n"])+row["force_b_abs_n"],0,atol=1e-10)
                    loaded+=row["force_a_abs_n"] if row["body_a"]=="loaded" else row["force_b_abs_n"]
                np.testing.assert_allclose(loaded,[-1,19.62,0],atol=.1)

    def test_force_sampling_preserves_all_solver_poses_and_marks_initial_unsolved(self):
        trace=self.traces["fixed"]
        self.assertEqual(801,len(trace["samples"]))
        expected=np.linspace(0,800,720,dtype=int).tolist()
        self.assertEqual(expected,trace["force_evidence_sampling"]["solver_indices"])
        self.assertFalse(trace["samples"][0]["force_evidence"]["available"])
        self.assertEqual("unsolved",trace["samples"][0]["force_evidence"]["reason"])
        selected=set(expected)
        for index,frame in enumerate(trace["samples"][1:],start=1):
            evidence=frame["force_evidence"]
            self.assertEqual(index in selected,evidence["available"])
            if evidence["available"]:
                self.assertAlmostEqual((index-1)*.001,evidence["step_start_s"])
                self.assertAlmostEqual(index*.001,evidence["step_end_s"])
        self.assertEqual(self.scenes["fixed"]["joints"],trace["connections"])
        self.assertEqual([0,-9.81,0],trace["gravity_m_s2"])

    def test_initially_touching_source_meshes_remain_supported_in_both_orders(self):
        for name in ("touching_0","touching_1"):
            with self.subTest(name=name):
                trace=self.traces[name]
                self.assertIsNone(trace["failure"])
                self.assertEqual(51,len(trace["samples"]))
                heights=np.array(trace["signals"]["loaded.position.y_m"],copy=True)
                self.assertLessEqual(float(np.max(np.abs(heights-.01))),.00005)
                rows=trace["samples"][-1]["force_evidence"]["contacts"]
                self.assertTrue(rows)
                force=np.zeros(3)
                for row in rows:
                    force+=row["force_a_abs_n"] if row["body_a"]=="loaded" else row["force_b_abs_n"]
                np.testing.assert_allclose(force,[0,19.62,0],atol=.1)

    def test_touching_support_balances_gravity_from_the_first_step(self):
        for name in ("touching_0","touching_1"):
            with self.subTest(name=name):
                frame=self.traces[name]["samples"][1]
                self.assertGreaterEqual(frame["body_poses"]["loaded"]["position_abs_m"][1],.01-1e-10)
                force=np.zeros(3)
                for row in frame["force_evidence"]["contacts"]:
                    force+=row["force_a_abs_n"] if row["body_a"]=="loaded" else row["force_b_abs_n"]
                np.testing.assert_allclose(force,[0,19.62,0],atol=1e-5)

    def test_penetration_search_window_detects_overshot_source_mesh(self):
        from ..slim_harness.runtime import CONTACT

        trace=self.traces["overshot_60um"]
        failure=trace["failure"]
        self.assertIsNotNone(failure,"A source box 60 micrometres inside its floor must be detected")
        self.assertEqual("contact_penetration",failure["kind"])
        self.assertGreater(failure["max_penetration_m"],CONTACT["penetration_tolerance_m"])
        self.assertEqual(.00005,CONTACT["penetration_tolerance_m"])

    def test_coplanar_roof_contact_keeps_support_without_false_front_penetration(self):
        trace=self.traces["coplanar_roof_front_corner"]
        self.assertIsNone(trace["failure"])
        self.assertEqual(11,len(trace["samples"]))
        heights=np.array(trace["signals"]["loaded.position.z_m"],copy=True)
        self.assertLessEqual(float(np.max(np.abs(heights-.005))),.00005)
        contacts=trace["samples"][-1]["force_evidence"]["contacts"]
        self.assertTrue(contacts)
        force=np.zeros(3)
        for contact in contacts:
            force+=contact["force_a_abs_n"] if contact["body_a"]=="loaded" else contact["force_b_abs_n"]
        np.testing.assert_allclose(force,[0,0,19.62],atol=.1)

    def test_shared_footprint_penetration_remains_detected_with_different_edge_subdivision(self):
        for name in ("shared_footprint_body","shared_footprint_subdivided_body"):
            with self.subTest(name=name):
                failure=self.traces[name]["failure"]
                self.assertIsNotNone(failure)
                self.assertEqual("contact_penetration",failure["kind"])
                self.assertGreater(failure["max_penetration_m"],.00005)
        # Confirm a retained deep contact at an added bottom-edge vertex whose
        # projection lies within an unsubdivided edge on the other source box.
        name="shared_footprint_subdivided_body"
        scene=self.scenes[name]
        loaded=next(b for b in scene["bodies"] if b["id"]=="loaded")
        floor=next(b for b in scene["bodies"] if b["id"]=="floor")
        loaded_mesh=trimesh.load(loaded["mesh_file"],force="mesh",process=False)
        floor_mesh=trimesh.load(floor["mesh_file"],force="mesh",process=False)
        retained_edge_contact=False
        for contact in self.traces[name]["failure"]["pairs"]:
            if contact["distance_m"]>=-.00005:continue
            loaded_side="a" if contact["body_a"]=="loaded" else "b"
            floor_side="b" if loaded_side=="a" else "a"
            local_loaded=np.asarray(contact[f"point_{loaded_side}_m"])-loaded["position_m"]
            local_floor=np.asarray(contact[f"point_{floor_side}_m"])-floor["position_m"]
            vertex_distances=np.linalg.norm(loaded_mesh.vertices-local_loaded,axis=1)
            vertex=loaded_mesh.vertices[int(np.argmin(vertex_distances))]
            on_bottom=abs(vertex[1]+.01)<1e-10
            on_one_side=(abs(abs(vertex[0])-.02)<1e-10) != (abs(abs(vertex[2])-.03)<1e-10)
            if (vertex_distances.min()<1e-8 and on_bottom and on_one_side
                    and np.linalg.norm(floor_mesh.vertices-local_floor,axis=1).min()>1e-8):
                retained_edge_contact=True
                break
        self.assertTrue(retained_edge_contact,"A real contact at a subdivided edge must not be discarded")

    def test_reflex_corner_keeps_its_real_horizontal_penetration_contact(self):
        failure=self.traces["l_corner_side_penetration"]["failure"]
        self.assertIsNotNone(failure)
        self.assertEqual("contact_penetration",failure["kind"])
        self.assertGreater(failure["max_penetration_m"],.00005)
        retained_horizontal_corner=False
        for contact in failure["pairs"]:
            if contact["distance_m"]>=-.00005:continue
            if {contact["body_a"],contact["body_b"]}!={"l_boundary","slider"}:continue
            corner_side="a" if contact["body_a"]=="l_boundary" else "b"
            other_side="b" if corner_side=="a" else "a"
            corner=np.asarray(contact[f"point_{corner_side}_m"])
            projected=np.asarray(contact[f"point_{other_side}_m"])
            delta=projected-corner
            if (np.linalg.norm(corner-[0,-.005,0])<1e-8
                    and abs(delta[0]+.00006)<1e-8
                    and abs(delta[1])<1e-8 and abs(delta[2])<1e-8):
                retained_horizontal_corner=True
                break
        self.assertTrue(retained_horizontal_corner,
                        "The L reflex vertex must retain its actual horizontal 60 um contact")

if __name__=="__main__":unittest.main()
