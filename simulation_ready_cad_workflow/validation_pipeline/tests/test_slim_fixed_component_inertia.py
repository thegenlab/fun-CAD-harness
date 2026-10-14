"""Controller inertia regressions; no native dynamics or source geometry mutation."""
import unittest
from types import SimpleNamespace
import numpy as np
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime import _effective_inertia
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.fixed_component_filter import build_fixed_component_filter

class Vec:
    def __init__(self,*v):self.x,self.y,self.z=map(float,v)
def array(v):return np.array([v.x,v.y,v.z])
class Body:
    def __init__(self,pos=(0,0,0),fixed=False,rotation=None):self.pos=np.array(pos,dtype=float);self.fixed=fixed;self.rotation=np.eye(3) if rotation is None else np.array(rotation)
    def IsFixed(self):return self.fixed
    def GetPos(self):return Vec(*self.pos)
    def GetCollisionModel(self):return None
    def TransformDirectionParentToLocal(self,v):return Vec(*(self.rotation.T@array(v)))
    def TransformPointLocalToParent(self,v):return Vec(*(self.pos+self.rotation@array(v)))
class Link:
    def __init__(self):self.disabled=False
    def IsDisabled(self):return self.disabled

class FixedComponentInertiaTests(unittest.TestCase):
    def setUp(self):
        self.c=SimpleNamespace(ChVector3d=Vec)
        self.bodies={'a':Body(),'b':Body(),'sat':Body((3,0,0)),'world':Body(fixed=True)}
        self.records={k:{'position_m':v.pos.tolist(),'mass_kg':m,'inertia_kg_m2':np.diag(i).tolist()} for (k,v),(m,i) in zip(self.bodies.items(),[(2,[2,4,6]),(3,[3,6,9]),(1,[1,2,3]),(20,[20,20,20])])}
        self.spec={'body_a':'a','body_b':'b','type':'revolute','anchor':[0,0,0]}
    def value(self,components=()):return _effective_inertia(self.c,self.bodies,self.records,self.spec,np.array([0.,1.,0.]),fixed_components=components)
    def test_unconnected_bodies_retain_original_estimate(self):self.assertAlmostEqual(self.value(),2.4)
    def test_empty_components_do_not_enumerate_unrelated_bodies(self):
        class NoEnumeration(dict):
            def __iter__(self):raise AssertionError("unrelated body enumeration")
        self.bodies=NoEnumeration(self.bodies)
        self.assertAlmostEqual(self.value(),2.4)
    def test_group_mass_inertia_and_parallel_axis(self):
        self.assertAlmostEqual(self.value([['a','sat']]),30/7)
        self.spec['type']='prismatic';self.assertAlmostEqual(self.value([['a','sat']]),1.5)
    def test_actual_body_orientation_is_used(self):
        self.bodies['sat'].rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        self.assertAlmostEqual(self.value([['a','sat']]),4.2)
    def test_component_connected_to_world_is_immovable(self):
        self.assertAlmostEqual(self.value([['a','sat'],['b','world']]),15.)
        self.spec['type']='prismatic';self.assertAlmostEqual(self.value([['a','sat'],['b','world']]),3.)
    def test_disabled_or_released_fixed_chain_refreshes(self):
        link=Link();row={'spec':{'type':'fixed','body_a':'a','body_b':'sat'},'active':True,'link':link}
        filt=build_fixed_component_filter(self.bodies,{'join':row})
        self.assertAlmostEqual(self.value(filt.plan['components']),30/7)
        link.disabled=True;self.assertTrue(filt.refresh(time_s=.001));self.assertAlmostEqual(self.value(filt.plan['components']),2.4)
        link.disabled=False;filt.refresh(time_s=.002);self.assertAlmostEqual(self.value(filt.plan['components']),30/7)
        row['active']=False;self.assertTrue(filt.refresh(time_s=.003));self.assertAlmostEqual(self.value(filt.plan['components']),2.4)
    def test_internal_fixed_connection_has_no_relative_response(self):self.assertEqual(self.value([['a','b']]),1e14)

if __name__=='__main__':unittest.main()
