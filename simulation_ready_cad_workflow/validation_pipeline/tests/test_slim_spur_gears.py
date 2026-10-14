"""Source geometry, passive native transmission, and finite engagement checks."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import numpy as np
import trimesh

from ..slim_harness.spur_gears import build_spur_gears, compile_spur_gears, SpurGearError
from .test_slim_equivalent_elasticity import Body, Vector


def nominal():
    return {"id":"mesh", "body_a":"a", "body_b":"b", "joint_a":"bearing_a", "joint_b":"bearing_b",
            "center_a_m":[0.,0.,0.], "center_b_m":[.08,0.,0.], "axis_abs":[0.,0.,1.],
            "teeth_a":20, "teeth_b":60, "module_m":.002, "face_width_m":.008,
            "pressure_angle_rad":math.radians(20), "stiffness_n_m":1e6, "damping_n_s_m":40.,
            "center_tolerance_m":.0002, "axis_tolerance_rad":.01, "parameter_basis":"Declared nominal test model"}


def tooth_mesh(teeth):
    angles = np.arange(teeth*4)*2*math.pi/(teeth*4)
    radius = np.tile([.002*(teeth/2-1.25), .002*(teeth/2+1), .002*(teeth/2+1), .002*(teeth/2-1.25)],teeth)
    ring = np.c_[radius*np.cos(angles),radius*np.sin(angles)]
    vertices=np.array([[x,y,z] for z in (-.004,.004) for x,y in ring]+[[0,0,-.004],[0,0,.004]])
    n=len(ring);faces=[]
    for i in range(n):
        j=(i+1)%n
        faces.extend([[i,j,j+n],[i,j+n,i+n],[2*n,j,i],[2*n+1,i+n,j+n]])
    return trimesh.Trimesh(vertices=vertices,faces=faces,process=False)


class SourceSpurGearTests(unittest.TestCase):
    def setUp(self):
        runs=Path(__file__).resolve().parents[3]/".runtime_runs/spur_gear_tests"
        runs.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=runs);self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);spec=nominal();records=[]
        for side in ("a","b"):
            path=root/f"{side}.obj";tooth_mesh(spec[f"teeth_{side}"]).export(path)
            records.append({"id":side,"fixed":False,"position_m":spec[f"center_{side}_m"],
                            "rotation_wxyz":[1,0,0,0],"mesh_file":str(path),"source_solid_ids":["solid_1"]})
        records.append({"id":"frame","fixed":True})
        self.geometry={"bodies":records}
        self.scene={"units":"m","spur_gears":[spec],"joints":[
            {"id":"bearing_"+side,"body_a":side,"body_b":"frame","type":"revolute",
             "anchor":spec[f"center_{side}_m"],"axis":[0,0,1]} for side in ("a","b")]}

    def test_source_teeth_and_native_coordinate_binding(self):
        row=compile_spur_gears(self.scene,self.geometry)[0]
        self.assertEqual(row["source_evidence"]["a"]["observed_teeth"],20)
        self.assertEqual(row["source_evidence"]["b"]["observed_teeth"],60)
        np.testing.assert_allclose(row["center_b_local_m"],[0,0,0])

    def test_explicit_compliant_provider_can_replace_an_ideal_bearing(self):
        scene = copy.deepcopy(self.scene)
        scene["equivalent_elasticity"] = {"elements": [
            {"id": joint["id"], "kind": "shaft_support", "body_a": joint["body_a"],
             "body_b": "frame", "axis_abs": joint["axis"], "anchor_abs_m": joint["anchor"]}
            for joint in scene["joints"]]}
        scene["joints"] = []
        row = compile_spur_gears(scene, self.geometry)[0]
        self.assertEqual(row["joint_a"], "bearing_a")
        scene["equivalent_elasticity"]["elements"][0]["axis_abs"] = [1,0,0]
        with self.assertRaisesRegex(SpurGearError, "center/axis"):
            compile_spur_gears(scene, self.geometry)

    def test_wrong_source_profile_is_rejected(self):
        tooth_mesh(19).export(self.geometry["bodies"][0]["mesh_file"])
        with self.assertRaises(SpurGearError):compile_spur_gears(self.scene,self.geometry)

    def test_cap_interior_vertices_do_not_change_the_source_tooth_profile(self):
        for side in ("a","b"):
            mesh=tooth_mesh(self.scene["spur_gears"][0][f"teeth_{side}"]).subdivide().subdivide()
            mesh.export(next(b["mesh_file"] for b in self.geometry["bodies"] if b["id"]==side))
        row=compile_spur_gears(self.scene,self.geometry)[0]
        self.assertEqual(row["source_evidence"]["a"]["observed_teeth"],20)
        self.assertEqual(row["source_evidence"]["b"]["observed_teeth"],60)

    def test_axially_separated_thin_gears_cannot_claim_a_wider_common_face(self):
        for side, shift in (("a", -.002), ("b", .002)):
            mesh=tooth_mesh(self.scene["spur_gears"][0][f"teeth_{side}"])
            mesh.vertices[:,2]=mesh.vertices[:,2]*.125+shift
            mesh.export(next(b["mesh_file"] for b in self.geometry["bodies"] if b["id"]==side))
        with self.assertRaisesRegex(SpurGearError,"axial face interval"):
            compile_spur_gears(self.scene,self.geometry)

    def test_split_tooth_bands_do_not_prove_continuous_face_material(self):
        pieces=[]
        for shift in (-.0035,.0035):
            mesh=tooth_mesh(20);mesh.vertices[:,2]=mesh.vertices[:,2]*.125+shift;pieces.append(mesh)
        trimesh.util.concatenate(pieces).export(self.geometry["bodies"][0]["mesh_file"])
        with self.assertRaisesRegex(SpurGearError,"axial face interval"):
            compile_spur_gears(self.scene,self.geometry)

    def test_compound_shaft_can_have_another_gear_outside_declared_face(self):
        extra=tooth_mesh(60);extra.apply_translation([0,0,.018])
        trimesh.util.concatenate([tooth_mesh(20),extra]).export(self.geometry["bodies"][0]["mesh_file"])
        row=compile_spur_gears(self.scene,self.geometry)[0]
        self.assertEqual(row["source_evidence"]["a"]["checked_axial_tooth_sections"],20)

    def test_wrong_bearing_axis_and_disengaged_centers_are_rejected(self):
        for change in ("axis","distance"):
            scene=copy.deepcopy(self.scene)
            if change=="axis":scene["joints"][0]["axis"]=[1,0,0]
            else:scene["spur_gears"][0]["center_b_m"][0]+=.003
            with self.assertRaises(SpurGearError):compile_spur_gears(scene,self.geometry)

    def test_negative_damping_is_rejected(self):
        self.scene["spur_gears"][0]["damping_n_s_m"]=-1
        with self.assertRaises(SpurGearError):compile_spur_gears(self.scene,self.geometry)

    def test_high_contact_ratio_source_is_in_the_supported_model(self):
        spec=self.scene["spur_gears"][0]
        spec.update(teeth_a=100,teeth_b=100,pressure_angle_rad=math.radians(14.5),center_b_m=[.2,0.,0.])
        self.scene["joints"][1]["anchor"]=[.2,0.,0.]
        for record in self.geometry["bodies"][:2]:
            record["position_m"]=spec[f'center_{record["id"]}_m']
            tooth_mesh(100).export(record["mesh_file"])
        self.assertEqual(100,compile_spur_gears(self.scene,self.geometry)[0]["source_evidence"]["b"]["observed_teeth"])


class SpurFrictionImpulseTests(unittest.TestCase):
    def test_multiple_stations_and_shared_pairs_cannot_add_free_body_kinetic_energy(self):
        # One known preceding normal reaction isolates the actual friction law
        # from the native tooth solve; full source inertia and all application
        # points still determine the resulting discrete rigid-body impulse.
        class FreeBody(Body):
            inertia=np.asarray([[.004,.001,0.],[.001,.006,0.],[0.,0.,.005]])
            def GetMass(self):return 1.
            def GetInertiaXX(self):return Vector(*np.diag(self.inertia))
            def GetInertiaXY(self):return Vector(.001,0.,0.)
            def GetRot(self):return SimpleNamespace(e0=1.,e1=0.,e2=0.,e3=0.)
        class SolvedGear:
            def __getattr__(self,name):
                if name=="IsTeethCompliant":return lambda:True
                if name=="GetContactForce":return lambda:1.
                if name.startswith("GetRotation"):return lambda:0.
                return lambda *args:None
        for pair_count,separate in ((1,False),(3,False),(2,True)):
            with self.subTest(pair_count=pair_count,separate=separate):
                bodies={"a":FreeBody([0,0,0],np.eye(3))};rows=[];joints={}
                for index in range(pair_count):
                    key=f"b{index}";angle=0. if separate else index*2*math.pi/3
                    first=f"a{index}" if separate and index else "a"
                    offset=.4*index if separate else 0.
                    if first not in bodies:bodies[first]=FreeBody([offset,0,0],np.eye(3))
                    bodies[key]=FreeBody([offset+.2*math.cos(angle),.2*math.sin(angle),0.],np.eye(3))
                    spec=nominal();spec.update(id=f"mesh{index}",body_a=first,body_b=key,joint_a=first,joint_b=key,
                        teeth_a=100,teeth_b=100,pressure_angle_rad=math.radians(14.5),
                        center_a_local_m=[0,0,0],center_b_local_m=[0,0,0],
                        axis_a_local=[0,0,1],axis_b_local=[0,0,1])
                    rows.append(spec);joints[spec["joint_a"]]={"active":True};joints[key]={"active":True}
                direction=np.asarray([-math.cos(math.radians(14.5)),math.sin(math.radians(14.5)),0.])
                for key,body in bodies.items():
                    sign=1 if key.startswith("a") else -1
                    speed=.1 if separate and key in {"a1","b1"} else 1e-5
                    body.velocity=sign*speed*direction
                    body.omega=np.asarray([2e-6,-3e-6,1e-5])*sign
                c=SimpleNamespace(ChLinkLockGear=SolvedGear,NarrowphaseCallback=object,ChVector3d=Vector,ChFramed=lambda *args:None)
                system=MagicMock();system.GetChTime.return_value=0.
                handle=build_spur_gears(c,system,bodies,joints,rows,axis_quat=lambda *args:None,friction=.3)
                dt=.001
                def energy(after,keys=None):
                    total=0.
                    for key in (keys if keys is not None else bodies):
                        body=bodies[key]
                        velocity=body.velocity+dt*body.force if after else body.velocity
                        omega=body.omega+dt*np.linalg.solve(body.inertia,body.torque) if after else body.omega
                        total+=.5*float(velocity@velocity)+.5*float(omega@body.inertia@omega)
                    return total
                before=energy(False);slow_before=energy(False,["a","b0"]);handle.apply_friction(dt)
                self.assertEqual(6*pair_count,len(handle.friction_records))
                self.assertLess(energy(True),before)
                if separate:self.assertLess(energy(True,["a","b0"]),slow_before)
                self.assertLess(handle.minimum_friction_impulse_scale,1.)
                np.testing.assert_allclose(sum((b.force for b in bodies.values()),np.zeros(3)),0.,atol=1e-15)
                np.testing.assert_allclose(sum((b.torque+np.cross(b.position,b.force) for b in bodies.values()),np.zeros(3)),0.,atol=1e-15)
                for row in handle.friction_records:
                    self.assertLessEqual(np.linalg.norm(row["force_abs_n"]),.3/3+1e-14)


CHRONO = Path(os.environ.get("SIMREADY_TEST_CHRONO_PYTHON",
    str(Path.home() / ".conda/envs/chrono_py312_projectchrono/python.exe")))


@unittest.skipUnless(CHRONO.is_file(), "Native Chrono interpreter not configured")
class NativeSpurGearTests(unittest.TestCase):
    def test_runtime_uses_gears_without_the_optional_source_contact_modules(self):
        # A source-profile fixture tests runtime wiring, not assembly support
        # or whole-machine acceptance. Formal original-solid audit is separate.
        fixture=SourceSpurGearTests("test_source_teeth_and_native_coordinate_binding")
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        folder=Path(fixture.temp.name)
        prepared={"gravity_m_s2":[0,0,0],"bodies":copy.deepcopy(fixture.geometry["bodies"]),
                  "joints":copy.deepcopy(fixture.scene["joints"]),
                  "spur_gears":compile_spur_gears(fixture.scene,fixture.geometry)}
        trimesh.creation.box(extents=[.01,.01,.01]).export(folder/"frame.obj")
        for body in prepared["bodies"]:
            if body["id"]=="frame":
                body.update(position_m=[0,-.2,0],rotation_wxyz=[1,0,0,0],
                            mesh_file=str(folder/"frame.obj"),source_solid_ids=["solid_1"])
            body.update(mass_kg=.1,inertia_kg_m2=np.diag([.0001]*3).tolist())
        positions={b["id"]:np.asarray(b["position_m"]) for b in prepared["bodies"]}
        for joint in prepared["joints"]:
            for side in ("a","b"):
                local=(np.asarray(joint["anchor"])-positions[joint[f"body_{side}"]]).tolist()
                joint[f"region_{side}"]={"points":[local,local],"radius":.002}
        test={"id":"gear_wiring","duration_s":.02,"step_s":.0001,
              "inputs":[{"joint":"bearing_a","mode":"effort","value":.02,"max_effort":.02}],
              "assertions":[{"id":"complete","expression":"last(signal('time_s')) >= 0.02"}]}
        output=folder/"trace.json";request=folder/"request.json"
        request.write_text(json.dumps({"scene":prepared,"test":test,"output":str(output)}))
        result=subprocess.run([str(CHRONO),"-m","simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",str(request)],
                              cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr+result.stdout[-2000:])
        from ..slim_harness.trace_storage import open_trace,release_trace_cache
        trace=open_trace(output)
        try:
            self.assertTrue(trace["physical_pass"],trace.get("failure"))
            self.assertIn("spur_gears.py",trace["execution_implementation"])
            self.assertIn("equivalent spur",trace["spur_gears"]["scope"])
            gear=trace["samples"][-1]["spur_gears"]["mesh"]
            self.assertGreater(abs(gear["rotation_a_rad"]),.001)
            self.assertAlmostEqual(gear["rotation_b_rad"]/gear["rotation_a_rad"],-1/3,delta=.025)
        finally:release_trace_cache(trace)

    def test_native_finite_transmission_passivity_release_and_local_contact_scope(self):
        source=r'''
import json,math
from types import SimpleNamespace
import numpy as np
from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import load_pychrono_core
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.spur_gears import build_spur_gears
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime import _axis_quat
c=load_pychrono_core();s=c.ChSystemNSC();s.SetGravitationalAcceleration(c.ChVector3d(0,0,0))
s.SetCollisionSystemType(c.ChCollisionSystem.Type_BULLET)
s.SetSolverType(c.ChSolver.Type_PSOR);s.GetSolver().AsIterative().SetMaxIterations(200);s.GetSolver().AsIterative().SetTolerance(0)
ground=c.ChBody();ground.SetFixed(True);s.Add(ground);bodies={};joints={}
for name,x in [('a',0.),('b',.08)]:
 b=c.ChBody();b.SetName(name);b.SetMass(.1);b.SetInertiaXX(c.ChVector3d(.0001,.0001,.0001));b.SetPos(c.ChVector3d(x,0,0));s.Add(b);bodies[name]=b
 j=c.ChLinkLockRevolute();j.Initialize(b,ground,c.ChFramed(c.ChVector3d(x,0,0)));s.AddLink(j);joints['bearing_'+name]={'active':True,'link':j}
row={'id':'mesh','body_a':'a','body_b':'b','joint_a':'bearing_a','joint_b':'bearing_b','center_a_local_m':[0,0,0],'center_b_local_m':[0,0,0],
 'axis_a_local':[0,0,1],'axis_b_local':[0,0,1],'teeth_a':20,'teeth_b':60,'module_m':.002,'face_width_m':.008,
 'pressure_angle_rad':math.radians(20),'stiffness_n_m':1e6,'damping_n_s_m':40.,'center_tolerance_m':.0002,'axis_tolerance_rad':.01}
h=build_spur_gears(c,s,bodies,joints,[row],axis_quat=_axis_quat,friction=.3)
acc=bodies['a'].AddAccumulator();dt=.0001;work=0.
for _ in range(50):h.begin_step();h.apply_friction(dt);s.Update();s.DoStepDynamics(dt)
assert abs(bodies['a'].GetAngVelParent().z)<1e-10 and abs(bodies['b'].GetAngVelParent().z)<1e-10
for _ in range(2000):
 before=bodies['a'].GetAngVelParent().z;bodies['a'].EmptyAccumulator(acc);bodies['a'].AccumulateTorque(acc,c.ChVector3d(0,0,.02),False)
 h.begin_step();h.apply_friction(dt);s.Update();s.DoStepDynamics(dt);work+=.02*.5*(before+bodies['a'].GetAngVelParent().z)*dt
wa,wb=[bodies[k].GetAngVelParent().z for k in ('a','b')]
energy=.5*.0001*(wa*wa+wb*wb)
assert wa>0 and wb<0 and abs(wb/wa+1/3)<.015,(wa,wb)
assert energy<=work*1.02+1e-8,(energy,work)
assert h.friction_energy_j>=0 and h.entries[0]['link'].IsTeethCompliant()
def model(key):return SimpleNamespace(GetContactable=lambda:SimpleNamespace(GetPhysicsItem=lambda:SimpleNamespace(GetName=lambda:key)))
# Scope probes use the actual current body-frame transforms, after rotation.
for b in bodies.values():b.SetRot(c.ChQuaterniond(1,0,0,0))
info=SimpleNamespace(modelA=model('a'),modelB=model('b'),vpA=c.ChVector3d(.02,0,0),vpB=c.ChVector3d(.02,0,0))
assert h.OnNarrowphase(info) is False
info.modelB=model('third_party');assert h.OnNarrowphase(info) is True
info.modelB=model('b')
info.vpA=info.vpB=c.ChVector3d(.02,0,.02);assert h.OnNarrowphase(info) is True
joints['bearing_b']['active']=False;joints['bearing_b']['link'].SetDisabled(True)
bodies['b'].SetPos(c.ChVector3d(.12,0,0));h.begin_step();h.apply_friction(dt)
assert not h.entries[0]['active'] and h.sample()['mesh']['normal_force_n']==0 and not h.friction_records
info.vpA=info.vpB=c.ChVector3d(.02,0,0);assert h.OnNarrowphase(info) is True
released_speed=bodies['b'].GetAngVelParent().z
for _ in range(50):h.begin_step();h.apply_friction(dt);s.Update();s.DoStepDynamics(dt)
assert abs(bodies['b'].GetAngVelParent().z-released_speed)<1e-8
print(json.dumps({'wa':wa,'wb':wb,'kinetic_energy_j':energy,'input_work_j':work,'passive_friction_j':h.friction_energy_j,'released':True}))
'''
        result=subprocess.run([str(CHRONO),"-c",source],cwd=Path(__file__).resolve().parents[3],
            capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr+result.stdout)
        self.assertTrue(json.loads(result.stdout.strip().splitlines()[-1])["released"])
