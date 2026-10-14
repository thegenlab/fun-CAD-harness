"""Reject a proved outward CE vertex constraint on an original convex star.

This is not an overlap test. A deep interior material witness keeps the contact
for rejection; non-mesh shapes, compound bodies and unresolved geometry stay.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path

import numpy as np


NATIVE_COORDINATE_BAND_M = 1e-8
DIRECTION_DOT_BAND = 1e-8


class _ClosedSourceMesh:
    def __init__(self, record):
        self.path = str(record['mesh_file'])
        self.valid_single_boundary = False
        self.star_cache = {}
        self.supported_vertex_face_cache = {}
        if len(record.get('source_solid_ids', [])) != 1:
            return
        vertices, faces = [], []
        for line in Path(self.path).read_text(encoding='utf-8').splitlines():
            values = line.split()
            if values and values[0] == 'v':
                vertices.append([float(value) for value in values[1:4]])
            elif values and values[0] == 'f':
                if len(values) != 4:
                    return
                faces.append([int(value.split('/')[0])-1 for value in values[1:]])
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        if not vertices or not faces:
            return
        self.triangles = self.vertices[self.faces]
        if not np.isfinite(self.triangles).all():
            return
        self.triangle_min = self.triangles.min(axis=1)
        self.triangle_max = self.triangles.max(axis=1)
        self.face_query_roundoff = 64*np.finfo(float).eps*max(
            float(np.max(np.abs(self.triangle_min))), float(np.max(np.abs(self.triangle_max))), 1.)
        self.edge_a = self.triangles[:, 1]-self.triangles[:, 0]
        self.edge_b = self.triangles[:, 2]-self.triangles[:, 0]
        cross = np.cross(self.edge_a, self.edge_b)
        self.double_areas = np.linalg.norm(cross, axis=1)
        if np.any(self.double_areas <= 0):
            return
        self.normals = cross/self.double_areas[:, None]
        self.incident = [[] for _ in self.vertices]
        edges = {}
        for index, face in enumerate(self.faces):
            for vertex in face:
                self.incident[vertex].append(index)
            for a, b in zip(face, np.roll(face, -1)):
                edges.setdefault(tuple(sorted((int(a), int(b)))), []).append((int(a), int(b), index))
        if not all(len(rows) == 2 and rows[0][:2] == rows[1][:2][::-1] for rows in edges.values()):
            return
        volume = np.einsum('ij,ij->i', self.triangles[:, 0],
                           np.cross(self.triangles[:, 1], self.triangles[:, 2])).sum()/6
        if not math.isfinite(float(volume)) or volume <= 0:
            return
        adjacency = [[] for _ in self.faces]
        for rows in edges.values():
            first, second = rows[0][2], rows[1][2]
            adjacency[first].append(second)
            adjacency[second].append(first)
        seen, pending = set(), [0]
        while pending:
            index = pending.pop()
            if index not in seen:
                seen.add(index)
                pending.extend(adjacency[index])
        self.valid_single_boundary = len(seen) == len(self.faces)

    def nearest_vertex(self, point):
        distance = np.linalg.norm(self.vertices-point, axis=1)
        index = int(np.argmin(distance))
        return index, float(distance[index])

    def face_candidates(self, point, band):
        padding = band+self.face_query_roundoff
        return np.flatnonzero(np.all(point >= self.triangle_min-padding, axis=1)
                              & np.all(point <= self.triangle_max+padding, axis=1))

    def supported_vertex_faces(self, vertex):
        if vertex not in self.supported_vertex_face_cache:
            incident = self.incident[vertex]
            supported = []
            if incident:
                delta = self.vertices[np.unique(self.faces[incident])]-self.vertices[vertex]
                guard = 64*np.finfo(float).eps*float(np.max(np.linalg.norm(delta, axis=1)))
                for face in incident:
                    outside = float(np.max(delta@self.normals[face]))
                    if outside <= guard:
                        supported.append((face, outside, guard))
            self.supported_vertex_face_cache[vertex] = tuple(supported)
        return self.supported_vertex_face_cache[vertex]

    def star(self, vertex):
        if vertex in self.star_cache:
            return self.star_cache[vertex]
        faces = self.incident[vertex]
        neighbors = np.unique(self.faces[faces])
        neighbors = neighbors[neighbors != vertex]
        delta = self.vertices[neighbors]-self.vertices[vertex]
        direction = delta.mean(axis=0)
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length <= NATIVE_COORDINATE_BAND_M:
            result = {'clear_convex': False}
        else:
            direction /= length
            normals = self.normals[faces]
            max_outside = float(np.max(delta@normals.T))
            max_dot = float(np.max(normals@direction))
            # This geometry is original float64 OBJ data. Do not use the
            # native contact-coordinate band to forgive a concave star.
            halfspace_guard = 64*np.finfo(float).eps*float(np.max(np.linalg.norm(delta, axis=1)))
            result = {
                'clear_convex': max_outside <= halfspace_guard and max_dot < -DIRECTION_DOT_BAND,
                'max_neighbor_outside_m': max_outside,
                'star_halfspace_roundoff_guard_m': halfspace_guard,
                'max_normal_dot_d': max_dot,
                'd_local': direction.tolist(),
                'incident_face_indices': faces,
                'neighbor_vertex_indices': neighbors.tolist(),
            }
        self.star_cache[vertex] = result
        return result

    def material(self, point):
        """Closed, consistently oriented single-boundary material, without CAD."""
        if not np.isfinite(point).all():
            return {'state': 'unknown'}
        a, b, c = (self.triangles[:, index] for index in range(3))
        signed_plane = np.einsum('ij,ij->i', point-a, self.normals)
        projected_relative = point-signed_plane[:, None]*self.normals-a
        u = np.einsum('ij,ij->i', np.cross(projected_relative, self.edge_b), self.normals)/self.double_areas
        v = np.einsum('ij,ij->i', np.cross(self.edge_a, projected_relative), self.normals)/self.double_areas
        on_triangle = (u >= 0) & (v >= 0) & (u+v <= 1)
        distance2 = np.where(on_triangle, signed_plane*signed_plane, np.inf)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = second-first
            t = np.clip(np.einsum('ij,ij->i', point-first, edge)/np.einsum('ij,ij->i', edge, edge), 0, 1)
            delta = point-first-t[:, None]*edge
            distance2 = np.minimum(distance2, np.einsum('ij,ij->i', delta, delta))
        distance = math.sqrt(float(np.min(distance2)))
        if not math.isfinite(distance):
            return {'state': 'unknown'}
        if distance <= NATIVE_COORDINATE_BAND_M:
            return {'state': 'surface_band', 'boundary_distance_m': distance}
        va, vb, vc = a-point, b-point, c-point
        la, lb, lc = (np.linalg.norm(value, axis=1) for value in (va, vb, vc))
        numerator = np.einsum('ij,ij->i', va, np.cross(vb, vc))
        denominator = (la*lb*lc + np.einsum('ij,ij->i', va, vb)*lc
                       + np.einsum('ij,ij->i', vb, vc)*la + np.einsum('ij,ij->i', vc, va)*lb)
        winding = float(np.arctan2(numerator, denominator).sum()/(2*math.pi))
        state = ('inside' if abs(abs(winding)-1) < 1e-5 else
                 'outside' if abs(winding) < 1e-5 else 'unknown')
        return {'state': state, 'boundary_distance_m': distance, 'winding_number': winding}


def build_source_contact_filter(c, system, bodies, records, *, penetration_tolerance_m, previous_filter=None):
    sources = {key: _ClosedSourceMesh(record) for key, record in records.items()}
    # Chrono 10 releases use MESHTRIANGLE; newer builds renamed the same type.
    ce_triangle_type = (c.ChCollisionShape.CONNECTEDTRIANGLE
                        if hasattr(c.ChCollisionShape, "CONNECTEDTRIANGLE")
                        else c.ChCollisionShape.MESHTRIANGLE)

    class SourceContactFilter(c.NarrowphaseCallback):
        def __init__(self):
            super().__init__()
            self.source_meshes = sources
            self.counts = Counter()
            self.rejected_samples = []
            self.owner_poses = {}

        def begin_step(self):
            self.owner_poses.clear()
            if previous_filter is not None:
                previous_filter.begin_step()

        def owner_pose(self, key):
            if key not in self.owner_poses:
                body = bodies[key]
                p, q = body.GetPos(), body.GetRot()
                quaternion = np.asarray([q.e0, q.e1, q.e2, q.e3], dtype=float)
                quaternion /= np.linalg.norm(quaternion)
                w, x, y, z = quaternion
                rotation = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
                ])
                self.owner_poses[key] = rotation, np.array([p.x, p.y, p.z])
            return self.owner_poses[key]

        def keep(self, reason):
            self.counts[reason] += 1
            return True

        def OnNarrowphase(self, info):
            try:
                return self.process_contact(info)
            except Exception:
                import traceback
                traceback.print_exc()
                raise

        def process_contact(self, info):
            self.counts['candidates'] += 1
            if previous_filter is not None and not previous_filter.OnNarrowphase(info):
                self.counts['previous_filter_rejections'] += 1
                return False
            if info.distance >= 0:
                return self.keep('nonnegative_kept')
            self.counts['negative_candidates'] += 1
            if info.shapeA.GetType() != ce_triangle_type or info.shapeB.GetType() != ce_triangle_type:
                return self.keep('non_CE_shapes_kept')
            keys = [str(model.GetContactable().GetPhysicsItem().GetName()) for model in (info.modelA, info.modelB)]
            if not all(key in sources and sources[key].valid_single_boundary for key in keys):
                return self.keep('compound_or_unknown_boundary_kept')
            points = [np.array([point.x, point.y, point.z]) for point in (info.vpA, info.vpB)]
            checks = []
            for side in (0, 1):
                own, other = keys[side], keys[1-side]
                rotation, position = self.owner_pose(own)
                index, error = sources[own].nearest_vertex((points[side]-position)@rotation)
                if error > NATIVE_COORDINATE_BAND_M:
                    continue
                star = sources[own].star(index)
                if not star['clear_convex']:
                    return self.keep('nonconvex_or_unknown_star_kept')
                vertex = sources[own].vertices[index]
                world = vertex@rotation.T+position
                direction = (points[1-side]-world)@rotation
                length = float(np.linalg.norm(direction))
                projection = float(direction@np.asarray(star['d_local']))
                guard = 2*NATIVE_COORDINATE_BAND_M+DIRECTION_DOT_BAND*length
                contradiction = math.isfinite(projection) and math.isfinite(guard) and projection < -guard
                dot = projection/length if length > NATIVE_COORDINATE_BAND_M else None
                checks.append({'vertex_body': own, 'other_body': other, 'vertex_index': index,
                               'source_mesh_file': sources[own].path, 'match_error_m': error,
                               'source_vertex_local_m': vertex.tolist(), 'source_vertex_world_m': world.tolist(),
                               'star': star, 'f_dot_d': dot,
                               'compression_projection_m': projection, 'contradiction_guard_m': guard,
                               'clear_outward_contradiction': contradiction})
            if not checks:
                return self.keep('no_matched_vertex_kept')
            if not any(check['clear_outward_contradiction'] for check in checks):
                return self.keep('no_outward_contradiction_kept')
            # A contradictory convex-star direction is necessary before the
            # expensive material query. Query every matched end: any genuine
            # deep interior witness or unresolved query preserves the contact.
            # A shallow overlap does not validate a contradictory normal with
            # a much larger reported depth. Valid inward normals were already
            # kept above, including their ordinary small numerical overlaps.
            for check in checks:
                rotation, position = self.owner_pose(check['other_body'])
                point = (np.asarray(check['source_vertex_world_m'])-position)@rotation
                material = sources[check['other_body']].material(point)
                self.counts['material_queries'] += 1
                check['other_material'] = material
                if (material['state'] == 'inside'
                        and material['boundary_distance_m']+NATIVE_COORDINATE_BAND_M >= penetration_tolerance_m):
                    return self.keep('deep_material_inside_kept')
                if material['state'] == 'unknown':
                    return self.keep('material_unknown_kept')
            self.counts['rejected'] += 1
            if len(self.rejected_samples) < 16:
                self.rejected_samples.append({
                    'time_s': float(system.GetChTime()), 'body_pair': keys,
                    'distance_m': float(info.distance), 'points_world_m': [point.tolist() for point in points],
                    'reason': 'convex_vertex_compression_direction_contradiction', 'endpoint_checks': checks,
                })
            return False

        def metadata(self):
            return {
                'schema_version': 'original_source_convex_star_contact_filter_v1',
                'rule': 'Reject only a negative CE candidate with a matched convex vertex, a strictly inward star direction, a contradictory compression direction, and no matched vertex providing a deep material witness at the unchanged penetration limit',
                'deep_material_keep_limit_m': penetration_tolerance_m,
                'material_distance_scope': 'Distance of this vertex to the other closed mesh boundary; not a bound on whole-body overlap and not a replacement for original STEP audit',
                'native_coordinate_band_m': NATIVE_COORDINATE_BAND_M,
                'direction_dot_band': DIRECTION_DOT_BAND,
                'coordinate_band_role': 'Native float coordinate uncertainty; not an acceptance penetration tolerance',
                'contradiction_projection_guard': '2*native_coordinate_band_m + direction_dot_band*contact_segment_length_m',
                'star_halfspace_roundoff_guard': '64*float64_epsilon*maximum_local_star_edge_length_m',
                'native_connected_triangle_enum': int(ce_triangle_type),
                'counts': dict(self.counts),
                'cached_vertex_stars': sum(len(source.star_cache) for source in sources.values()),
                'maximum_saved_rejections': 16,
                'rejected_candidate_samples': deepcopy(self.rejected_samples),
                'previous_filter_chained': previous_filter is not None,
                'geometry_mass_inertia_joints_changed': False,
            }

    callback = SourceContactFilter()
    system.GetCollisionSystem().RegisterNarrowphaseCallback(callback)
    return callback
