"""Focused candidate checks; no native solver or BRep audit is invoked."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,unittest
import numpy as np
from ..slim_harness import source_capsule_contacts as module


def cube_mesh():
    vertices=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    triangles=vertices[faces]; edge_a=triangles[:,1]-triangles[:,0]; edge_b=triangles[:,2]-triangles[:,0]
    cross=np.cross(edge_a,edge_b); double_areas=np.linalg.norm(cross,axis=1)
    incident=[[i for i,face in enumerate(faces) if vertex in face] for vertex in range(len(vertices))]
    mesh=SimpleNamespace(vertices=vertices,faces=faces,triangles=triangles,edge_a=edge_a,edge_b=edge_b,normals=cross/double_areas[:,None],
                         double_areas=double_areas,incident=incident,valid_single_boundary=True)
    mesh.face_candidates=lambda point,band:np.flatnonzero(np.all(point>=triangles.min(axis=1)-band-1e-14,axis=1)&np.all(point<=triangles.max(axis=1)+band+1e-14,axis=1))
    return mesh


def capsule_record():
    axis=np.array([.05,1.,0.]); axis/=np.linalg.norm(axis)
    cap={'type':'capsule','source_solid_id':'one','center_local_m':[1.2,0.,0.],'axis_source':axis.tolist(),'radius_m':.1,'straight_length_m':4.,
         'proof':{'cut_is_done':True,'missing_volume_mm3':0.,'original_solid_modified':False,
                  'method':'OCCT non-destructive Cut(complete_source_derived_cylinder, original_solid)'}}
    return {'source_solid_ids':['one'],'source_analytic_material':[cap]},cap


class Vector:
    def __init__(self,x,y,z):self.x,self.y,self.z=float(x),float(y),float(z)


def model(name):
    return SimpleNamespace(GetContactable=lambda:SimpleNamespace(GetPhysicsItem=lambda:SimpleNamespace(GetName=lambda:name)))


class FocusedTests(unittest.TestCase):
    def test_patch_boundary_is_local_and_preserves_original_edge_order(self):
        class NoGlobalScan(dict):
            def items(self):
                raise AssertionError("Patch lookup must not scan every source edge")
            def __iter__(self):
                raise AssertionError("Patch lookup must not scan every source edge")
        topology=module.SourcePatches(cube_mesh())
        topology.edges=NoGlobalScan(topology.edges)
        key=topology.patch(2)
        self.assertEqual([2,3],topology.patches[key]['faces'])
        self.assertEqual([(4,5),(5,6),(6,7),(4,7)],topology.patches[key]['boundary'])

    def test_finite_edge_not_infinite_plane(self):
        triangle=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        rows=module.nearest_segment_triangle(np.array([2.,2,-1]),np.array([2.,2,1]),triangle)
        self.assertAlmostEqual(min(r[0] for r in rows),np.sqrt(4.5),places=13)
        np.testing.assert_allclose(rows[0][2],[.5,.5,0],atol=1e-14)

    def test_crossing_and_degenerate_axis(self):
        triangle=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        rows=module.nearest_segment_triangle(np.array([.2,.2,-1]),np.array([.2,.2,1]),triangle)
        self.assertEqual(min(r[0] for r in rows),0.)
        self.assertIsNone(module.nearest_segment_triangle(np.zeros(3),np.zeros(3),triangle))

    def test_parallel_patch_keeps_location_ambiguity(self):
        topology=module.SourcePatches(cube_mesh()); patch,reason,near=topology.matched_patch(np.array([1.,0.,0.]))
        self.assertIsNone(reason)
        result,reason=topology.nearest(np.array([[1.2,-.5,0],[1.2,.5,0]]),.1,patch)
        self.assertIsNone(result); self.assertEqual(reason,'nonunique_contact_location')

    def test_near_edge_and_multiple_patches(self):
        topology=module.SourcePatches(cube_mesh())
        patch,reason,near=topology.matched_patch(np.array([1.,1.+1e-9,0.]))
        self.assertIsNone(patch); self.assertEqual(reason,'different_or_unresolved_patch'); self.assertTrue(near)
        self.assertIn(6,near)

    def test_convex_cone_and_inconsistent_incident_normals(self):
        mesh=cube_mesh(); topology=module.SourcePatches(mesh); key=topology.patch(6)
        point=np.array([1.,-1.,0.]); normal=np.array([1.,-.1,0.]); normal/=np.linalg.norm(normal)
        support,reason=topology.boundary_support(point,normal,topology.patches[key])
        self.assertIsNone(reason); self.assertTrue(support['normal_cone']['in_cone_roundoff'])
        for face in support['incident_faces']:
            if mesh.normals[face,1]<-.9:mesh.normals[face]*=-1
        support,reason=topology.boundary_support(point,normal,topology.patches[key])
        self.assertIsNone(support); self.assertEqual(reason,'nonconvex_source_boundary')
        self.assertFalse(module.normal_cone(np.eye(3),np.array([-1.,0.,0.]))['in_cone_roundoff'])

    def test_capsule_creation_identity_and_positive_length(self):
        record,cap=capsule_record()
        self.assertIsNotNone(module.eligible_capsule(record,[cap]))
        self.assertIsNone(module.eligible_capsule(record,[]))
        self.assertIsNone(module.eligible_capsule(record,[cap,cap]))
        cap['straight_length_m']=0.
        self.assertIsNone(module.eligible_capsule(record,[cap]))

    def test_wrapper_ab_symmetry_chain_and_step_cache(self):
        record,cap=capsule_record(); mesh=cube_mesh()
        positions={'wall':np.zeros(3),'axle':np.zeros(3)}
        source=SimpleNamespace(source_meshes={'wall':mesh,'axle':mesh},owner_pose=lambda key:(np.eye(3),positions[key]))
        class Previous:
            def __init__(self):self.steps=0; self.calls=[]; self.keep=True
            def begin_step(self):self.steps+=1
            def OnNarrowphase(self,info):self.calls.append(info.distance); return self.keep
            def metadata(self):return {'previous':'present'}
        previous=Previous(); registered=[]
        c=SimpleNamespace(ChCollisionShape=SimpleNamespace(CONNECTEDTRIANGLE=11,CAPSULE=7),NarrowphaseCallback=object,ChVector3d=Vector)
        system=SimpleNamespace(GetChTime=lambda:0.,GetCollisionSystem=lambda:SimpleNamespace(RegisterNarrowphaseCallback=registered.append))
        cb=module.build_source_capsule_contacts(c,system,source,{'axle':record},previous,created_capsules_by_body={'axle':[cap]})
        def contact(reverse=False):
            sides=[('wall',11,[1.,0.,0.]),('axle',7,[1.5,0.,0.])]
            if reverse:sides.reverse()
            return SimpleNamespace(modelA=model(sides[0][0]),modelB=model(sides[1][0]),shapeA=SimpleNamespace(GetType=lambda:sides[0][1]),
                 shapeB=SimpleNamespace(GetType=lambda:sides[1][1]),vpA=Vector(*sides[0][2]),vpB=Vector(*sides[1][2]),vN=Vector(-1 if reverse else 1,0,0),distance=.5)
        cb.begin_step(); a=contact(); self.assertTrue(cb.OnNarrowphase(a)); b=contact(True); self.assertTrue(cb.OnNarrowphase(b))
        self.assertEqual(cb.counts['reconstructed'],2); self.assertEqual(cb.counts['finite_patch_queries'],1)
        self.assertAlmostEqual(a.distance,b.distance,places=14)
        np.testing.assert_allclose([a.vpA.x,a.vpA.y,a.vpA.z],[b.vpB.x,b.vpB.y,b.vpB.z],atol=1e-14)
        np.testing.assert_allclose([a.vN.x,a.vN.y,a.vN.z],-np.array([b.vN.x,b.vN.y,b.vN.z]),atol=1e-14)
        previous.keep=False; positions['axle']=np.array([.05,0.,0.]); cb.begin_step()
        moved=contact(); self.assertFalse(cb.OnNarrowphase(moved))
        self.assertGreater(moved.distance,a.distance+.049)
        self.assertLess(moved.distance,a.distance+.051)
        self.assertEqual(previous.steps,2); self.assertEqual(cb.counts['finite_patch_queries'],2)
        self.assertEqual(len(previous.calls),3); self.assertTrue(cb.metadata()['previous_callback_chained'])

if __name__=='__main__':unittest.main()
