"""Source-face reconstruction on closed boxes; no native solver or CAD claim."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from ..slim_harness import source_contact_normals as module
from ..slim_harness.source_contact_filter import _ClosedSourceMesh, NATIVE_COORDINATE_BAND_M


class Vector:
    def __init__(self,x,y,z):self.x,self.y,self.z=map(float,(x,y,z))


def xyz(vector):return np.array([vector.x,vector.y,vector.z])


def model(name):
    return SimpleNamespace(GetContactable=lambda:SimpleNamespace(
        GetPhysicsItem=lambda:SimpleNamespace(GetName=lambda:name)))


BOX_FACES=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                    [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])


class SourceContactNormalTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.positions={"A":np.zeros(3),"B":np.zeros(3)}
        self.rotations={"A":np.eye(3),"B":np.eye(3)}
        self.chained=[]
        self.keep=True
        self.step_count=0

    def box(self,name,minimum,maximum):
        lo,hi=np.asarray(minimum,float),np.asarray(maximum,float)
        vertices=np.array([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],
            [hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
            [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],
            [hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]])
        path=self.root/(name+".obj")
        path.write_text("\n".join([*("v "+" ".join(format(x,".17g") for x in row) for row in vertices),
            *("f "+" ".join(str(int(x)+1) for x in row) for row in BOX_FACES)])+"\n",encoding="utf-8")
        mesh=_ClosedSourceMesh({"mesh_file":str(path),"source_solid_ids":[name]})
        self.assertTrue(mesh.valid_single_boundary)
        return mesh

    def callback(self,a,b):
        self.sources={"A":a,"B":b}
        def begin():self.step_count+=1
        def chain(info):
            self.chained.append(self.snapshot(info))
            return self.keep
        source=SimpleNamespace(source_meshes=self.sources,
            owner_pose=lambda key:(self.rotations[key],self.positions[key]),
            begin_step=begin,OnNarrowphase=chain)
        self.registered=[]
        c=SimpleNamespace(ChCollisionShape=SimpleNamespace(CONNECTEDTRIANGLE=11),
            NarrowphaseCallback=object,ChVector3d=Vector)
        system=SimpleNamespace(GetChTime=lambda:float(self.step_count),GetCollisionSystem=lambda:SimpleNamespace(
            RegisterNarrowphaseCallback=self.registered.append))
        callback=module.build_source_contact_normals(c,system,source)
        self.assertEqual([callback],self.registered)
        callback.begin_step()
        return callback

    def shallow(self,depth=20.8e-9):
        a=self.box("A",[0,0,-depth],[.002,.002,.002-depth])
        b=self.box("B",[-.01,-.01,-.002],[.01,.01,0])
        return self.callback(a,b)

    def contact(self,point_a,point_b=None,*,length=.000155,normal=(-1,0,0),reverse=False):
        a=np.asarray(point_a,float)
        b=np.asarray(point_b,float) if point_b is not None else a+np.array([length,0,0])
        sides=[("A",a),("B",b)];normal=np.asarray(normal,float)
        if reverse:sides.reverse();normal=-normal
        return SimpleNamespace(modelA=model(sides[0][0]),modelB=model(sides[1][0]),
            shapeA=SimpleNamespace(GetType=lambda:11),shapeB=SimpleNamespace(GetType=lambda:11),
            vpA=Vector(*sides[0][1]),vpB=Vector(*sides[1][1]),vN=Vector(*normal),distance=-length)

    @staticmethod
    def snapshot(info):
        return (float(info.distance),tuple(xyz(info.vpA)),tuple(xyz(info.vpB)),tuple(xyz(info.vN)))

    def assert_unchanged(self,callback,info):
        before=self.snapshot(info)
        self.assertIsNone(callback.reconstruct(info))
        self.assertEqual(before,self.snapshot(info))

    def test_20_8nm_overlap_repairs_155um_lateral_native_candidate(self):
        cb=self.shallow();info=self.contact([0,0,-20.8e-9])
        change=cb.reconstruct(info)
        self.assertIsNotNone(change)
        self.assertAlmostEqual(-20.8e-9,info.distance,delta=1e-13)
        np.testing.assert_allclose(xyz(info.vN),[0,0,-1],atol=1e-13)
        np.testing.assert_allclose(xyz(info.vpA),[0,0,-20.8e-9],atol=1e-13)
        np.testing.assert_allclose(xyz(info.vpB),[0,0,0],atol=1e-13)

    def test_positive_source_gap_is_not_reconstructed_as_compression(self):
        cb=self.shallow(depth=-20.8e-9)
        self.assert_unchanged(cb,self.contact([0,0,20.8e-9]))

    def test_real_lateral_penetration_keeps_120um_depth(self):
        a=self.box("A",[-.00012,0,0],[.001,.002,.002])
        b=self.box("B",[-.01,-.01,-.01],[0,.01,.01])
        cb=self.callback(a,b)
        info=self.contact([-.00012,0,0],[-.00012,.000155,0],normal=(0,-1,0))
        self.assertIsNotNone(cb.reconstruct(info))
        self.assertAlmostEqual(-.00012,info.distance,delta=1e-13)
        self.assertLess(info.distance,-50e-6)
        np.testing.assert_allclose(xyz(info.vN),[-1,0,0],atol=1e-13)
        np.testing.assert_allclose(xyz(info.vpB),[0,0,0],atol=1e-13)

    def test_native_vertex_matching_remains_10nm(self):
        self.assertEqual(1e-8,NATIVE_COORDINATE_BAND_M)
        cb=self.shallow()
        self.assert_unchanged(cb,self.contact([11e-9,0,-20.8e-9]))
        near=self.contact([9e-9,0,-20.8e-9])
        self.assertIsNotNone(cb.reconstruct(near))
        np.testing.assert_allclose(xyz(near.vpA),[0,0,-20.8e-9],atol=1e-13)

    def test_target_query_uses_finite_triangle_not_plane_extension(self):
        # The top infinite plane is 20.8nm away, but its finite box is >200um away.
        a=self.box("A",[.0012,0,-20.8e-9],[.0015,.0003,.0003])
        b=self.box("B",[-.001,-.001,-.001],[.001,.001,0])
        cb=self.callback(a,b)
        self.assert_unchanged(cb,self.contact([.0012,0,-20.8e-9],length=.000155))

    def test_closer_unsupported_side_is_not_skipped_for_far_supported_top(self):
        # Source vertex has +x and -z incident normals. Nearest B side is +x,
        # only 20um away; the farther top is 100um away and would oppose -z.
        a=self.box("A",[.0005,0,-.0001],[.00098,.0003,.0002])
        b=self.box("B",[-.001,-.001,-.001],[.001,.001,0])
        cb=self.callback(a,b)
        self.assert_unchanged(cb,self.contact([.00098,0,-.0001],[.00098,.000155,-.0001]))

    def test_equal_nearest_normals_at_box_corner_are_ambiguous(self):
        p=20.8e-9
        a=self.box("A",[.001-p,0,-p],[.0013,.0003,.0003])
        b=self.box("B",[-.001,-.001,-.001],[.001,.001,0])
        cb=self.callback(a,b)
        self.assert_unchanged(cb,self.contact([.001-p,0,-p],[.001-p,.000155,-p]))

    def test_equal_depth_same_normal_at_distinct_endpoints_is_ambiguous(self):
        p=20.8e-9
        a=self.box("A",[-.0005,-.0005,-p],[.0015,.0015,.001])
        b=self.box("B",[-.001,-.001,-.001],[.001,.001,0])
        cb=self.callback(a,b)
        self.assert_unchanged(cb,self.contact([-.0005,-.0005,-p],[.001,.001,0],length=.003,normal=(0,0,-1)))

    def test_ab_symmetry_and_source_filter_chain_are_preserved(self):
        cb=self.shallow()
        a=self.contact([0,0,-20.8e-9]);b=self.contact([0,0,-20.8e-9],reverse=True)
        self.assertTrue(cb.OnNarrowphase(a));self.assertTrue(cb.OnNarrowphase(b))
        self.assertAlmostEqual(-20.8e-9,a.distance,delta=1e-13)
        self.assertAlmostEqual(a.distance,b.distance,delta=1e-13)
        np.testing.assert_allclose(xyz(a.vpA),xyz(b.vpB),atol=1e-13)
        np.testing.assert_allclose(xyz(a.vpB),xyz(b.vpA),atol=1e-13)
        np.testing.assert_allclose(xyz(a.vN),-xyz(b.vN),atol=1e-13)
        self.keep=False
        self.assertFalse(cb.OnNarrowphase(self.contact([0,0,-20.8e-9])))
        self.assertEqual(3,len(self.chained))

    def test_step_cache_rechecks_changed_owner_pose(self):
        cb=self.shallow();first=self.contact([0,0,-20.8e-9])
        self.assertIsNotNone(cb.reconstruct(first))
        self.positions["A"]=np.array([0,0,41.6e-9]);cb.begin_step()
        self.assert_unchanged(cb,self.contact([0,0,20.8e-9]))
        self.assertEqual(2,self.step_count)

    def test_same_step_short_long_short_radius_does_not_reuse_wrong_query(self):
        cb=self.shallow(depth=80e-9)
        self.assert_unchanged(cb,self.contact([0,0,-80e-9],length=1e-9))
        long=self.contact([0,0,-80e-9])
        self.assertIsNotNone(cb.reconstruct(long))
        self.assertAlmostEqual(-80e-9,long.distance,delta=1e-13)
        self.assert_unchanged(cb,self.contact([0,0,-80e-9],length=1e-9))


if __name__=="__main__":unittest.main()
