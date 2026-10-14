"""Automatic pictures and replay assets from the original body meshes."""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
import numpy as np
import trimesh
from ..mechanism.io import write_json_atomic

_RENDER_LOCK = threading.RLock()

def render_views(geometry: dict, output: Path, samples=None, *, exploded=False, focus=None):
    from ..benchmark_sandbox.visual_policy import require_nonvisual
    require_nonvisual("render_views")
    # Matplotlib's figure/font state is process-global. Physics and authors
    # stay parallel; only this short plotting section is serialized.
    with _RENDER_LOCK:
        return _render_views(geometry,output,samples,exploded=exploded,focus=focus)


def _render_views(geometry: dict, output: Path, samples=None, *, exploded=False,focus=None):
    from ..benchmark_sandbox.visual_policy import require_nonvisual
    require_nonvisual("render_views")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    output.parent.mkdir(parents=True,exist_ok=True)
    records = geometry["bodies"]
    scope = geometry.get("validation_scope", {})
    reference_ids = set(scope.get("elastic_reference_body_ids", []))
    meshes,selected_records = [],[]
    for index,body in enumerate(records):
        mesh = trimesh.load_mesh(body["mesh_file"],process=False)
        pose = samples[body["id"]] if samples else {"position_abs_m":body["position_m"],"rotation_abs_quat_wxyz":body["rotation_wxyz"]}
        matrix = trimesh.transformations.quaternion_matrix(pose["rotation_abs_quat_wxyz"])
        matrix[:3,3] = pose["position_abs_m"]
        mesh.apply_transform(matrix)
        if focus:
            center=np.asarray(focus["center_m"]);radius=float(focus["radius_m"])
            triangles=mesh.triangles
            mask=np.all(triangles.max(axis=1)>=center-radius,axis=1)&np.all(triangles.min(axis=1)<=center+radius,axis=1)
            mesh=mesh.submesh([np.flatnonzero(mask)],append=True,repair=False)
            if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces): continue
            # Clip the visible surface, rather than retain a long triangle
            # whose vertices would make a close-up fit the whole machine.
            for dimension in range(3):
                normal=np.zeros(3);normal[dimension]=1
                lower=center.copy();lower[dimension]-=radius
                upper=center.copy();upper[dimension]+=radius
                mesh=trimesh.intersections.slice_mesh_plane(mesh,normal,lower,cap=False)
                mesh=trimesh.intersections.slice_mesh_plane(mesh,-normal,upper,cap=False)
            if not len(mesh.faces): continue
        meshes.append(mesh)
        selected_records.append(body)
    if not meshes: raise ValueError("Requested close-up contains no original geometry")
    records=selected_records
    allpoints = np.vstack([m.vertices for m in meshes])
    size = np.ptp(allpoints,axis=0)
    if exploded:
        center = allpoints.mean(axis=0)
        for i,mesh in enumerate(meshes):
            direction = mesh.centroid-center
            if np.linalg.norm(direction)<1e-8: direction=np.array([0,0,1.])
            mesh.apply_translation(direction/np.linalg.norm(direction)*max(size)*(.25+i*.02))
        allpoints = np.vstack([m.vertices for m in meshes])
    minimum,maximum = allpoints.min(axis=0),allpoints.max(axis=0)
    center = (minimum+maximum)/2
    span = np.maximum(maximum-minimum,max(maximum-minimum)*.15)
    fig = plt.figure(figsize=(14,16 if focus else 11),facecolor="white")
    palettes = [(0.40,.54,.65),(.82,.64,.26),(.55,.63,.53),(.69,.42,.34),(.50,.43,.63)]
    views = [(23,-58,"assembled perspective"),(0,-90,"front"),(0,0,"side"),(90,-90,"top")]
    if focus: views=[(23,-58,"perspective"),(0,-90,"front"),(0,90,"back"),(0,0,"side"),(0,180,"opposite side"),(90,-90,"top")]
    if focus and ("elevation_deg" in focus or "azimuth_deg" in focus):
        elevation=float(focus.get("elevation_deg",23));azimuth=float(focus.get("azimuth_deg",-58))
        if not np.isfinite([elevation,azimuth]).all() or abs(elevation)>89 or abs(azimuth)>180:
            raise ValueError("Invalid evidence camera angles")
        views[0]=(elevation,azimuth,"requested oblique view")
    light = np.array([.4,-.3,.85]);light/=np.linalg.norm(light)
    for index,(elev,azim,title) in enumerate(views,1):
        ax = fig.add_subplot(3 if focus else 2,2,index,projection="3d")
        opaque_triangles,opaque_colors=[],[]
        for j,(body,mesh) in enumerate(zip(records,meshes)):
            raw = np.asarray(body.get("color") or palettes[j%len(palettes)],dtype=float)[:3]
            if raw.max()>1: raw/=255
            shade = np.clip(.7+.3*(mesh.face_normals@light),.35,1)
            is_reference = body["id"] in reference_ids
            if is_reference: raw = np.asarray([.55,.55,.55])
            colors = shade[:,None]*raw
            if is_reference:
                ax.add_collection3d(Poly3DCollection(mesh.triangles*1000,facecolors=colors,edgecolors="none",alpha=.28))
            else:
                opaque_triangles.append(mesh.triangles*1000);opaque_colors.append(colors)
            if focus:
                edges=mesh.face_adjacency_edges[mesh.face_adjacency_angles>.25]
                if len(edges):
                    ax.add_collection3d(Line3DCollection(mesh.vertices[edges]*1000,colors=[(.07,.09,.1,.8)],linewidths=.35))
        # Sort opaque faces across bodies. Per-body collections can paint a
        # whole large frame over a nearer pointer merely because one of the
        # frame's distant faces determines the collection depth.
        if opaque_triangles:
            ax.add_collection3d(Poly3DCollection(np.concatenate(opaque_triangles),
                facecolors=np.concatenate(opaque_colors),edgecolors="none",alpha=1))
        for axis,lo,hi in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center-span*.56,center+span*.56): axis(lo*1000,hi*1000)
        ax.set_box_aspect(span.copy())
        ax.view_init(elev=elev,azim=azim)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.set_title(title if not exploded else title+" / exploded diagram",fontsize=11)
    heading = "Original geometry"+(" — cropped connection close-up" if focus else " — exploded diagram (not a physical pose)" if exploded else "")
    if reference_ids: heading += "\nGray translucent springs: undeformed references, not solved deformation"
    fig.suptitle(heading,fontsize=15)
    fig.text(.03,.02," | ".join(b["id"] for b in records),fontsize=8)
    fig.savefig(output,dpi=135,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return output


def export_replay(geometry: dict, trace: dict, output: Path):
    from ..benchmark_sandbox.visual_policy import require_nonvisual
    require_nonvisual("export_replay")
    output.mkdir(parents=True,exist_ok=True)
    scene = trimesh.Scene()
    scope = trace.get("validation_scope", geometry.get("validation_scope", {}))
    reference_ids = set(scope.get("elastic_reference_body_ids", []))
    bodies = []
    for body in geometry["bodies"]:
        if body["id"] in reference_ids: continue
        mesh = trimesh.load_mesh(body["mesh_file"],process=False)
        color = np.asarray(body.get("color") or [160,174,186,255])
        if color.max() <= 1: color = np.rint(color*255).astype(np.uint8)
        mesh.visual.face_colors = color
        node = f"physics__{body['id']}__original"
        transform = trimesh.transformations.quaternion_matrix(body["rotation_wxyz"])
        transform[:3,3] = body["position_m"]
        scene.add_geometry(mesh,node_name=node,geom_name=node,transform=transform)
        bodies.append({"body_id":body["id"],"fixed":body["fixed"],"source_solid_ids":body["source_solid_ids"],
                       "initial_center_abs_m":body["position_m"],"initial_rotation_abs_quat_wxyz":body["rotation_wxyz"],
                       "kinematics":"recorded_rigid_body_pose",
                       "components":[{"node":node,"solid_id":"original","representation":"original_source_surface","triangle_count":len(mesh.faces)}]})
    model = output/"original-body-local.glb"
    temporary=model.with_name(model.name+"."+uuid.uuid4().hex+".pending")
    try:
        temporary.write_bytes(trimesh.exchange.gltf.export_glb(scene))
        temporary.replace(model)
    finally:
        temporary.unlink(missing_ok=True)
    recorded_indices=trace.get("force_evidence_sampling",{}).get("solver_indices")
    indices = np.asarray(recorded_indices,dtype=int) if recorded_indices else np.unique(np.linspace(0,len(trace["samples"])-1,min(720,len(trace["samples"]))).astype(int))
    frames = [{**trace["samples"][i],"source_sample_index":int(i)} for i in indices]
    connections = trace.get("connections", [])
    for frame in frames:
        frame["joint_active"] = {}
    for joint in connections:
        # A disk-backed signal opens a memory map. Read each joint column
        # once for this export, then release it before loading the next one.
        values=trace.get("signals",{}).get(joint["id"]+".active")
        for frame,index in zip(frames,indices):
            frame["joint_active"][joint["id"]]=bool(values[index]) if values is not None and len(values) else True
        del values
    recorded_forces = [frame.get("force_evidence", {}) for frame in frames if frame.get("force_evidence", {}).get("available")]
    force_values = []
    torque_values = []
    for item in recorded_forces:
        for joint in item.get("joint_reactions", []):
            force_values.extend(np.linalg.norm(joint.get(key, [0,0,0])) for key in ("force_a_abs_n","force_b_abs_n"))
            torque_values.extend(np.linalg.norm(joint.get(key, [0,0,0])) for key in ("torque_a_abs_nm","torque_b_abs_nm"))
        for contact in item.get("contacts", []):
            force_values.extend(np.linalg.norm(contact.get(key, [0,0,0])) for key in ("force_a_abs_n","force_b_abs_n"))
        for load in item.get("external_loads", []):
            force_values.append(np.linalg.norm(load.get("force_abs_n", [0,0,0])))
            torque_values.append(np.linalg.norm(load.get("torque_abs_nm", [0,0,0])))
    replay = {"schema_version":"source_geometry_replay_v1","source_kind":"original_geometry",
              "branches":{"physical":{"branch":"physical","frames":frames,"status":"RECORDED",
                  "field_availability":{"joint_reactions":bool(recorded_forces),"contact_forces":bool(recorded_forces),
                                        "external_loads":bool(recorded_forces),"connection_regions":bool(connections)},
                  "maxima":{"force_n":float(max(force_values,default=0)),"torque_nm":float(max(torque_values,default=0))}}},
              "source_sample_count":len(trace["samples"]),"display_frame_count":len(frames),"landmarks":[],
              "validation_scope":scope,
              "connections":connections,
              "force_display":{"heatmap":"relative strength of recorded applied and reaction forces on original surfaces; not stress",
                               "paths":"schematic connections between force application points on the same rigid body; not material stress trajectories"},
              "claims":{"simulation_was_rerun_for_visuals":False,"rendered_geometry_is_original":True,"certification_authority":False,
                        "elastic_reference_geometry_is_deformation_evidence":False},
              "reference_geometry_note":"Undeformed elastic reference meshes are omitted from dynamic replay; their complete recorded root-attached poses remain in the trace. Elastic deformation was not solved." if reference_ids else None}
    minimum,maximum = np.full(3,np.inf),np.full(3,-np.inf)
    for body in geometry["bodies"]:
        if body["id"] in reference_ids: continue
        mesh = trimesh.load_mesh(body["mesh_file"],process=False)
        corners = trimesh.bounds.corners(mesh.bounds)
        for frame in frames:
            pose = frame["body_poses"][body["id"]]
            transform = trimesh.transformations.quaternion_matrix(pose["rotation_abs_quat_wxyz"])
            transform[:3,3] = pose["position_abs_m"]
            points = trimesh.transform_points(corners,transform)
            minimum=np.minimum(minimum,points.min(axis=0));maximum=np.maximum(maximum,points.max(axis=0))
    write_json_atomic(output/"replay.json",replay)
    write_json_atomic(output/"scene.json",{"schema_version":"source_geometry_scene_v1","source_kind":"original_geometry",
                      "bodies":bodies,"units":"m","component_vertices":"body_local_metres","validation_scope":scope,
                      "omitted_elastic_reference_body_ids":sorted(reference_ids),
                      "motion_bounds_abs_m":{"min":minimum.tolist(),"max":maximum.tolist()}})
    return {"model_file":str(model),"scene_file":str(output/"scene.json"),"replay_file":str(output/"replay.json")}
