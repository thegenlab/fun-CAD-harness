"""Optional machine-checkable shaft/bore evidence from the earlier native experiment runner.

The existing independent reviewer still checks every joint. This check rejects
false explicit support claims; absent declarations remain review obligations,
not automatic support certification or a requirement to match one topology.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .source_queries import unsigned_distance, contains_material

TOLERANCE_M=0.000025


def check_supports(scene,geometry):
    scale=.001 if scene["units"]=="mm" else 1.
    records={b["id"]:b for b in geometry["bodies"]}
    meshes={}
    def world_mesh(identifier):
        if identifier not in meshes:
            body=records[identifier]
            mesh=trimesh.load(body["mesh_file"],force="mesh",process=False)
            matrix=trimesh.transformations.quaternion_matrix(body["rotation_wxyz"])
            matrix[:3,3]=body["position_m"]
            mesh.apply_transform(matrix);meshes[identifier]=mesh
        return meshes[identifier]
    def vector(value):
        result=np.asarray(value,dtype=float)
        if result.shape!=(3,) or not np.isfinite(result).all():raise ValueError("Support needs finite xyz vectors")
        return result
    def cylinder(feature,kind):
        mesh=world_mesh(feature["body"])
        axis=vector(feature["axis"])
        if np.linalg.norm(axis)<1e-12:raise ValueError("Support axis is zero")
        axis/=np.linalg.norm(axis)
        center=vector(feature["center"])*scale
        radius,length=float(feature["radius"])*scale,float(feature["length"])*scale
        if not np.isfinite([radius,length]).all() or min(radius,length)<=0:
            raise ValueError("Support radius and length must be positive")
        u=np.cross(axis,np.eye(3)[np.argmin(abs(axis))]);u/=np.linalg.norm(u)
        v=np.cross(axis,u);angles=np.arange(32)*2*np.pi/32
        radial=np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
        points=np.concatenate([center+t*axis+radius*radial for t in np.linspace(-.5*length,.5*length,9)])
        if unsigned_distance(mesh,points).max()>TOLERANCE_M:
            raise ValueError(f"Declared {kind} surface is absent from the original source")
        offset=max(TOLERANCE_M*2,radius*.005)
        if radius<=offset:raise ValueError("Support feature below the validated geometric resolution")
        inward=np.concatenate([center+t*axis+(radius-offset)*radial for t in [-.25*length,.25*length]])
        outward=np.concatenate([center+t*axis+(radius+offset)*radial for t in [-.25*length,.25*length]])
        if kind=="shaft" and not contains_material(mesh,inward).all():
            raise ValueError("Shaft declaration has no source material")
        if kind=="bore" and (contains_material(mesh,inward).any() or not contains_material(mesh,outward).all()):
            raise ValueError("Bore is filled or has no surrounding source material")
        return center,axis,radius,length
    reports=[]
    for joint in scene.get("joints",[]):
        proof=joint.get("support")
        if proof is None:
            reports.append({"id":joint["id"],"status":"independent_review_required"});continue
        if joint["type"]!="revolute":raise ValueError("Explicit support currently implements shaft/bore revolute evidence only")
        shaft,bore=proof["shaft"],proof["bore"]
        owners={joint["body_a"],joint["body_b"]}
        if {shaft["body"],bore["body"]}!=owners:raise ValueError("Support evidence must belong to both joint bodies")
        ca,aa,ra,la=cylinder(shaft,"shaft");cb,ab,rb,lb=cylinder(bore,"bore")
        axis=vector(joint.get("axis",[0,0,1]));axis/=np.linalg.norm(axis)
        delta=cb-ca;anchor=vector(joint["anchor"])*scale
        gap=rb-ra;max_gap=float(proof["max_clearance"])*scale
        if (not np.isfinite(max_gap) or max_gap<0 or gap<0 or gap>max_gap
                or abs(aa@ab)<1-1e-7 or abs(axis@aa)<1-1e-7
                or np.linalg.norm(delta-aa*(delta@aa))>TOLERANCE_M
                or np.linalg.norm(anchor-ca-aa*((anchor-ca)@aa))>TOLERANCE_M
                or abs(delta@aa)>(la+lb)/2-min(la,lb)*.25):
            raise ValueError("Shaft/bore alignment, clearance or engagement does not support the declared joint")
        # A shaft in a hole proves radial support, not axial retention. Keep
        # retention as an explicit independent-review obligation; do not infer it.
        reports.append({"id":joint["id"],"status":"radial_support_checked",
                        "clearance_m":gap,"axial_retention":"independent_review_required"})
    return reports
