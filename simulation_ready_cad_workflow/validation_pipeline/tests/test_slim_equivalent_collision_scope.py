"""Portable classifier equivalence; synthetic metadata only, no CAD or native run."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.equivalent_collision_scope import build_equivalent_collision_scope
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.validation_scope import make_validation_scope, point_in_flexible_contact_region


def vec(p):return SimpleNamespace(x=float(p[0]),y=float(p[1]),z=float(p[2]))


class Body:
    def __init__(self,key):self.key=key;self.position=[.1,.2,.3];self.quaternion=[1.,0.,0.,0.];self.pose_reads=0
    def GetPos(self):return vec(self.position)
    def GetRot(self):self.pose_reads+=1;return SimpleNamespace(**dict(zip(('e0','e1','e2','e3'),self.quaternion)))
    def GetContactable(self):return self
    def GetPhysicsItem(self):return self
    def GetName(self):return self.key


class System:
    def GetCollisionSystem(self):return self
    def RegisterNarrowphaseCallback(self,callback):self.callback=callback


class EquivalentCollisionScopeTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name)
        self.records={key:{'id':key,'source_file':str(root/(key+'.step')),'source_sha256':'0'*64,
                          'source_solid_ids':['solid_1'],'position_m':[.1,.2,.3]} for key in ('a','b','other')}
        self.region={'id':'bore','body_pair':['a','b'],'region_owner_body':'a','shape':'cylinder',
                     'axis_origin_source_m':[0.,0.,0.],'axis_direction_source':[0.,0.,1.],
                     'radius_m':.00155,'axial_interval_m':[.0255,.029],
                     'source_refs':[{'path':self.records['a']['source_file'],'sha256':'0'*64,'locator':'synthetic classifier fixture'}]}
        self.bodies={key:Body(key) for key in self.records}

    def build(self,regions=None):
        scope=make_validation_scope(list(self.records.values()),flexible_contact_regions=regions or [self.region],model_binding={'fixture':True})
        return build_equivalent_collision_scope(SimpleNamespace(NarrowphaseCallback=object),System(),self.bodies,self.records,scope),scope

    def info(self,p,q,pair=('a','b')):
        return SimpleNamespace(modelA=self.bodies[pair[0]],modelB=self.bodies[pair[1]],vpA=vec(p),vpB=vec(q),distance=-.00001)

    def world(self,p):
        body=self.bodies['a'];w,x,y,z=np.asarray(body.quaternion)/np.linalg.norm(body.quaternion)
        r=np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        return np.asarray(body.position)+r@(np.asarray(p)-self.records['a']['position_m'])

    def test_moving_steps_reversed_pair_nonunit_quaternion_and_boundaries(self):
        callback,scope=self.build();r=self.region['radius_m'];lo,hi=self.region['axial_interval_m']
        for position,q in (([.1,.2,.3],[2.,0.,0.,0.]),([.2,-.1,.3],[1.5,.3,-.8,.2]),([-.4,.35,-.2],[.1,-.7,2.,.4])):
            self.bodies['a'].position=position;self.bodies['a'].quaternion=q;callback.begin_step()
            poses={'a':{'position_abs_m':position,'rotation_abs_quat_wxyz':q}}
            for radial in (0.,np.nextafter(r,0.),r,np.nextafter(r,np.inf),r+.001):
                for along in (np.nextafter(lo,-np.inf),lo,np.nextafter(lo,np.inf),(lo+hi)/2,hi,np.nextafter(hi,np.inf)):
                    a=self.world([radial,0.,along]);b=self.world([0.,0.,(lo+hi)/2])
                    for pair in (('a','b'),('b','a')):
                        expected=not all(point_in_flexible_contact_region(scope,pair,p,poses,
                            {'a':self.records['a']['position_m']}) for p in (a,b))
                        self.assertEqual(expected,callback.OnNarrowphase(self.info(a,b,pair)))
        self.assertEqual(3,self.bodies['a'].pose_reads)

    def test_same_region_both_points_counts_and_log_cap(self):
        other=copy.deepcopy(self.region);other.update(id='other_bore',axial_interval_m=[.0355,.039])
        callback,_=self.build([self.region,other]);callback.begin_step()
        a=self.world([0.,0.,.027]);b=self.world([0.,0.,.037])
        self.assertTrue(callback.OnNarrowphase(self.info(a,b)))
        for _ in range(24):self.assertFalse(callback.OnNarrowphase(self.info(a,a)))
        self.assertEqual({'bore':24,'other_bore':0},callback.excluded_counts);self.assertEqual(16,len(callback.last_exclusions))
        self.assertEqual(1,self.bodies['a'].pose_reads)
        callback.begin_step();self.assertEqual([],callback.last_exclusions)
        self.assertFalse(callback.OnNarrowphase(self.info(a,a)));self.assertEqual(25,callback.excluded_counts['bore'])
        self.assertEqual(2,self.bodies['a'].pose_reads)

    def test_unrelated_pair_never_reads_owner_pose(self):
        callback,_=self.build();callback.begin_step()
        self.assertTrue(callback.OnNarrowphase(self.info([0.,0.,.027],[0.,0.,.027],('a','other'))))
        self.assertEqual(0,self.bodies['a'].pose_reads);self.assertEqual([],callback.last_exclusions)


if __name__=='__main__':unittest.main()
