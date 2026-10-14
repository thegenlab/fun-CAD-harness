"""Source-material queries ported from the earlier native experiment runner (native_geometry.py).

Keep cavity winding and bound triangle-query memory. Results describe sampled
geometry, not a continuous Hausdorff or exact CAD certificate.
"""
from __future__ import annotations

import numpy as np
import trimesh


def unsigned_distance(mesh, points):
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points): return np.zeros(0)
    valid = np.asarray(mesh.area_faces) > 0
    origin = np.asarray(mesh.bounds).mean(axis=0)
    extent = float(np.ptp(mesh.vertices, axis=0).max())
    if not valid.any() or not np.isfinite(extent) or extent <= 0:
        raise ValueError("Source feature has no finite surface")
    scale = 1000.0 / extent
    query = trimesh.Trimesh((mesh.vertices-origin)*scale,mesh.faces[valid],process=False)
    # trimesh.closest_point rebuilds its referenced-vertex tree on every call.
    # Reuse it across the existing bounded batches, with the same guaranteed
    # candidate boxes and exact triangle closest-point computation.
    vertex_tree = cKDTree(query.vertices[query.referenced_vertices])
    triangle_tree = query.triangles_tree
    triangles = query.triangles.view(np.ndarray)
    result = np.empty(len(points))

    def distances(values, candidates):
        counts = np.asarray([len(row) for row in candidates],dtype=int)
        owners = np.repeat(np.arange(len(values)),counts)
        closest = trimesh.triangles.closest_point(triangles[np.concatenate(candidates)],values[owners])
        delta = values[owners]-closest
        squared = trimesh.util.diagonal_dot(delta,delta)
        starts = np.r_[0,np.cumsum(counts)[:-1]]
        return np.sqrt(np.minimum.reduceat(squared,starts))/scale

    for start in range(0,len(points),256):
        batch = min(256,len(points)-start)
        values = (points[start:start+batch]-origin)*scale
        radius = vertex_tree.query(values)[0].reshape(-1,1) + trimesh.constants.tol.merge
        boxes = np.column_stack((values-radius,values+radius))
        candidates, pair_count, group_start = [], 0, 0
        for index, box in enumerate(boxes):
            nearby = list(triangle_tree.intersection(box))
            if not nearby: raise ValueError("Source distance query has no candidate triangle")
            # Bound actual point/triangle pairs instead of assuming every
            # source face is near every point. A single large query retains
            # the same minimum-one-point behavior as the previous batching.
            if candidates and pair_count+len(nearby)>200000:
                result[start+group_start:start+index] = distances(values[group_start:index],candidates)
                candidates, pair_count, group_start = [], 0, index
            candidates.append(nearby)
            pair_count += len(nearby)
        result[start+group_start:start+batch] = distances(values[group_start:batch],candidates)
    if not np.isfinite(result).all(): raise ValueError("Nonfinite source distance")
    return result


def contains_material(mesh, points):
    # Split by face connectivity without trimesh.split's repair step, which can
    # turn an inward cavity into a positive solid. Winding is source evidence.
    parts = getattr(mesh,"_source_material_parts",None)
    if parts is None:
        groups = trimesh.graph.connected_components(mesh.face_adjacency,
            nodes=np.arange(len(mesh.faces)),min_len=1)
        parts = [mesh.submesh([faces],append=True,repair=False) for faces in groups]
        mesh._source_material_parts=parts
    positive = [p for p in parts if p.volume > 0]
    negative = [p for p in parts if p.volume < 0]
    points = np.asarray(points,dtype=float).reshape(-1,3)
    inside = np.zeros(len(points),dtype=bool)
    def contains(part,values):
        batch=max(1,min(256,200000//max(1,len(part.faces))))
        result=np.zeros(len(values),dtype=bool)
        for start in range(0,len(values),batch):result[start:start+batch]=part.contains(values[start:start+batch])
        return result
    for part in positive:
        occupied = contains(part,points)
        for cavity in negative:
            if (np.all(cavity.bounds[0]>=part.bounds[0]) and np.all(cavity.bounds[1]<=part.bounds[1])
                    and part.contains(cavity.vertices[:1])[0]):
                occupied &= ~contains(cavity,points)
        inside |= occupied
    return inside
