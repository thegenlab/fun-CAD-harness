"""Reconstruct CE contacts from nearest mutually supporting original faces.

Ambiguous directions, unsupported local faces, compound boundaries and genuine
source gaps keep their native contact. Reconstruction precedes the existing
source filter and leaves its equivalent-contact scope chain intact.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import numpy as np

from .source_contact_filter import NATIVE_COORDINATE_BAND_M


NORMAL_DOT_BAND = 1e-10
BARYCENTRIC_ROUNDOFF = 64*np.finfo(float).eps


def _nearest_faces(mesh, point, radius, indices=None):
    # The native contact segment bounds the search, not the accepted depth.
    # Compare finite triangles (including their edges) before support tests:
    # an unsupported nearer feature must never be skipped for a farther face.
    if indices is None:
        indices = mesh.face_candidates(point, radius)
    count = len(mesh.triangles)
    projected = np.empty((count, 3))
    plane = np.full(count, np.inf)
    u, v = np.zeros(count), np.zeros(count)
    projection_in_face = np.zeros(count, dtype=bool)
    if not len(indices):
        return indices, projected, projection_in_face, plane, u, v, math.inf, False
    a, b, c = (mesh.triangles[indices, index] for index in range(3))
    normals = mesh.normals[indices]
    local_plane = np.einsum('ij,ij->i', point-a, normals)
    local_projected = point-local_plane[:, None]*normals
    relative = local_projected-a
    local_u = np.einsum('ij,ij->i', np.cross(relative, mesh.edge_b[indices]), normals)/mesh.double_areas[indices]
    local_v = np.einsum('ij,ij->i', np.cross(mesh.edge_a[indices], relative), normals)/mesh.double_areas[indices]
    inside = (local_u >= 0) & (local_v >= 0) & (local_u+local_v <= 1)
    projection_in_face[indices] = ((local_u >= -BARYCENTRIC_ROUNDOFF) & (local_v >= -BARYCENTRIC_ROUNDOFF)
                                  & (local_u+local_v <= 1+BARYCENTRIC_ROUNDOFF))
    projected[indices], plane[indices] = local_projected, local_plane
    u[indices], v[indices] = local_u, local_v
    distance2 = np.where(inside, local_plane*local_plane, np.inf)
    closest = local_projected.copy()
    for first, second in ((a, b), (b, c), (c, a)):
        edge = second-first
        t = np.clip(np.einsum('ij,ij->i', point-first, edge)/np.einsum('ij,ij->i', edge, edge), 0, 1)
        edge_point = first+t[:, None]*edge
        delta = point-edge_point
        edge_distance2 = np.einsum('ij,ij->i', delta, delta)
        closer = edge_distance2 < distance2
        closest[closer] = edge_point[closer]
        distance2 = np.minimum(distance2, edge_distance2)
    distance = np.sqrt(distance2)
    minimum = float(np.min(distance))
    roundoff = mesh.face_query_roundoff
    if minimum > radius+roundoff:
        return indices[:0], projected, projection_in_face, plane, u, v, math.inf, False
    near = distance <= minimum+roundoff
    locations = closest[near]
    unique_location = bool(np.all(np.linalg.norm(locations-locations[0], axis=1) <= 4*roundoff))
    return indices[near], projected, projection_in_face, plane, u, v, minimum, unique_location


def build_source_contact_normals(c, system, source_filter):
    sources = source_filter.source_meshes
    coordinate_band = NATIVE_COORDINATE_BAND_M
    # Chrono 10 releases use MESHTRIANGLE; newer builds renamed the same type.
    ce_triangle_type = (c.ChCollisionShape.CONNECTEDTRIANGLE
                        if hasattr(c.ChCollisionShape, "CONNECTEDTRIANGLE")
                        else c.ChCollisionShape.MESHTRIANGLE)

    class SourceContactNormals(c.NarrowphaseCallback):
        def __init__(self):
            super().__init__()
            self.vertex_queries = {}
            self.face_candidate_cache = {}
            self.counts = Counter()
            self.changed_samples = []

        def begin_step(self):
            source_filter.begin_step()
            self.vertex_queries.clear()

        def vertex_matches(self, own, other, index, side, radius):
            cache_key = own, other, index, side
            previous = self.vertex_queries.get(cache_key)
            roundoff = sources[other].face_query_roundoff
            if previous is not None and (math.isfinite(previous[1]) or radius <= previous[0]):
                # Once an exhaustive radius search finds the nearest surface,
                # a larger radius cannot reveal a closer one. An empty short
                # search cannot answer a later, longer native contact.
                self.counts['vertex_query_cache_hits'] += 1
                return previous[2] if previous[1] <= radius+roundoff else []
            self.counts['vertex_queries'] += 1
            mesh, target = sources[own], sources[other]
            rotation, position = source_filter.owner_pose(own)
            other_rotation, other_position = source_filter.owner_pose(other)
            vertex = mesh.vertices[index]
            world = vertex@rotation.T+position
            other_local = (world-other_position)@other_rotation
            incident = mesh.supported_vertex_faces(index)
            cached = self.face_candidate_cache.get(cache_key)
            if cached is None or np.linalg.norm(other_local-cached[0])+radius > cached[2]:
                # Reuse only a provable candidate superset. Every current
                # nearest-point, support and normal test still runs each step.
                cached = (other_local.copy(), target.face_candidates(other_local, 2*radius), 2*radius)
                self.face_candidate_cache[cache_key] = cached
                self.counts['face_candidate_cache_refreshes'] += 1
            else:
                self.counts['face_candidate_cache_hits'] += 1
            near, projected, in_face, plane, u, v, minimum, unique_location = _nearest_faces(
                target, other_local, radius, cached[1])
            matches = []
            if len(near) and not unique_location:
                self.counts['nonunique_nearest_source_location'] += 1
            if len(near) and unique_location and incident:
                other_vertices = target.vertices[np.unique(target.faces[near])]
                supported_other = []
                for other_face in near:
                    if not in_face[other_face]:
                        continue
                    other_delta = other_vertices-target.triangles[other_face, 0]
                    other_guard = 64*np.finfo(float).eps*float(np.max(np.linalg.norm(other_delta, axis=1)))
                    other_outside = float(np.max(other_delta@target.normals[other_face]))
                    if other_outside <= other_guard:
                        supported_other.append((other_face, other_outside, other_guard))
                for own_face, own_outside, own_guard in incident:
                    own_normal = mesh.normals[own_face]
                    own_world_normal = own_normal@rotation.T
                    for other_face, other_outside, other_guard in supported_other:
                        other_normal = target.normals[other_face]
                        other_world_normal = other_normal@other_rotation.T
                        dot = float(own_world_normal@other_world_normal)
                        if dot > -1+NORMAL_DOT_BAND:
                            continue
                        projected_world = projected[other_face]@other_rotation.T+other_position
                        a, b = (world, projected_world) if side == 0 else (projected_world, world)
                        normal = -other_world_normal if side == 0 else other_world_normal
                        matches.append({
                            'vertex_body': own, 'other_body': other, 'vertex_index': index,
                            'own_source_mesh': mesh.path, 'other_source_mesh': target.path,
                            'own_face': int(own_face), 'other_face': int(other_face),
                            'source_vertex_world_m': world.tolist(),
                            'source_projection_world_m': projected_world.tolist(),
                            'source_plane_distance_m': float(plane[other_face]),
                            'nearest_source_boundary_distance_m': minimum,
                            'projected_barycentric_uv': [float(u[other_face]), float(v[other_face])],
                            'normal_dot': dot,
                            'own_max_outside_m': own_outside, 'own_roundoff_guard_m': float(own_guard),
                            'other_max_outside_m': other_outside, 'other_roundoff_guard_m': float(other_guard),
                            'point_a_world_m': a.tolist(), 'point_b_world_m': b.tolist(),
                            'normal_A_to_B_world': normal.tolist(),
                            'reconstructed_distance_m': float((b-a)@normal),
                        })
            self.vertex_queries[cache_key] = (radius, minimum, matches)
            return matches

        def unchanged(self, reason):
            self.counts[reason] += 1
            return None

        def reconstruct(self, info):
            self.counts['candidates'] += 1
            if info.distance > coordinate_band:
                return self.unchanged('positive_gap_unchanged')
            if info.shapeA.GetType() != ce_triangle_type or info.shapeB.GetType() != ce_triangle_type:
                return self.unchanged('non_CE_unchanged')
            keys = [str(model.GetContactable().GetPhysicsItem().GetName()) for model in (info.modelA, info.modelB)]
            if not all(key in sources and sources[key].valid_single_boundary for key in keys):
                return self.unchanged('compound_or_unknown_boundary_unchanged')
            points = [np.array([point.x, point.y, point.z]) for point in (info.vpA, info.vpB)]
            radius = float(np.linalg.norm(points[1]-points[0]))+2*coordinate_band
            matches = []
            endpoint_matches = []
            for side in (0, 1):
                own, other = keys[side], keys[1-side]
                rotation, position = source_filter.owner_pose(own)
                index, error = sources[own].nearest_vertex((points[side]-position)@rotation)
                if error > coordinate_band:
                    continue
                endpoint_matches.append({'vertex_body': own, 'vertex_index': index, 'match_error_m': error})
                matches.extend(self.vertex_matches(own, other, index, side, radius))
            groups = []
            for match in matches:
                normal = np.asarray(match['normal_A_to_B_world'])
                if not any(float(normal@other) > 1-NORMAL_DOT_BAND for other in groups):
                    groups.append(normal)
            if len(groups) != 1:
                return self.unchanged('no_unique_supported_source_normal')
            selected = matches[0]
            if any(max(np.linalg.norm(np.asarray(match[key])-selected[key])
                       for key in ('point_a_world_m', 'point_b_world_m')) > 2*coordinate_band for match in matches):
                return self.unchanged('ambiguous_source_contact_location')
            distance = selected['reconstructed_distance_m']
            if not math.isfinite(distance) or distance > coordinate_band:
                return self.unchanged('positive_source_gap_unchanged')
            before = {
                'distance_m': float(info.distance),
                'normal_A_to_B_world': [info.vN.x, info.vN.y, info.vN.z],
                'point_a_world_m': points[0].tolist(), 'point_b_world_m': points[1].tolist(),
            }
            after = {key: selected[key] for key in ('normal_A_to_B_world', 'point_a_world_m', 'point_b_world_m')}
            after['distance_m'] = distance
            if before == after:
                return self.unchanged('already_source_consistent')
            info.vN = c.ChVector3d(*after['normal_A_to_B_world'])
            info.vpA = c.ChVector3d(*after['point_a_world_m'])
            info.vpB = c.ChVector3d(*after['point_b_world_m'])
            info.distance = distance
            self.counts['reconstructed_candidates'] += 1
            if len(self.changed_samples) >= 16:
                return {}
            return {
                'time_s': float(system.GetChTime()), 'body_pair': keys,
                'reason': 'unique_mutually_supporting_source_faces',
                'before': before, 'after': after, 'endpoint_matches': endpoint_matches,
                'selected_source_evidence': selected, 'supporting_face_matches': len(matches),
                'distance_change_m': distance-before['distance_m'],
                'point_a_change_m': float(np.linalg.norm(np.asarray(after['point_a_world_m'])-points[0])),
                'point_b_change_m': float(np.linalg.norm(np.asarray(after['point_b_world_m'])-points[1])),
            }

        def OnNarrowphase(self, info):
            try:
                change = self.reconstruct(info)
                keep = source_filter.OnNarrowphase(info)
                if change is not None:
                    self.counts['reconstructed_kept' if keep else 'reconstructed_rejected_by_source_filter'] += 1
                    if change:
                        change['kept_after_source_filter'] = bool(keep)
                        self.changed_samples.append(change)
                return keep
            except Exception:
                import traceback
                traceback.print_exc()
                raise

        def metadata(self):
            return {
                'schema_version': 'original_source_nearest_contact_normals_v2',
                'rule': 'Reconstruct from a matched source vertex, a unique nearest finite source location, an in-face projection, mutually supporting local source planes and one opposite-normal group; retain actual signed depth',
                'native_coordinate_band_m': coordinate_band,
                'coordinate_band_role': 'Native endpoint/source vertex matching and roundoff-scale positive gap only; not a negative contact depth limit',
                'source_face_search_radius': 'Native endpoint separation plus twice the native coordinate band; bounds search only, never clips depth',
                'nearest_location_roundoff': '4 * source mesh face-query floating-point roundoff',
                'normal_dot_band': NORMAL_DOT_BAND,
                'projection_barycentric_roundoff': float(BARYCENTRIC_ROUNDOFF),
                'support_roundoff_guard': '64*float64_epsilon*maximum_local_neighbor_distance',
                'maximum_contact_location_disagreement_m': 2*coordinate_band,
                'distance_definition': '(source_point_b-source_point_a) dot normal_A_to_B',
                'counts': dict(self.counts),
                'vertex_query_cache_scope': 'Current solver step; original source meshes and current owner poses are shared with the source filter',
                'face_candidate_cache_scope': 'Only candidate indices persist; reuse requires anchor radius to contain current point plus current search radius; current nearest point and support tests always rerun',
                'maximum_saved_changes': 16,
                'changed_contact_samples': deepcopy(self.changed_samples),
                'source_filter_chained_after_reconstruction': True,
                'geometry_mass_inertia_joints_changed': False,
                'acceptance_penetration_tolerance_changed': False,
            }

    callback = SourceContactNormals()
    system.GetCollisionSystem().RegisterNarrowphaseCallback(callback)
    return callback
