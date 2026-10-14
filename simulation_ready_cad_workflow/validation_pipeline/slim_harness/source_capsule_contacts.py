"""Source-contained capsules against finite original coplanar patches.

The caller supplies records AND supplements actually instantiated by the runtime.
Multiple capsules, unresolved source geometry and nonunique contact locations retain
native contacts. The wrapper reconstructs before the existing callback chain.
"""
from collections import Counter
from copy import deepcopy
import itertools
import math
import numpy as np

from .source_contact_filter import NATIVE_COORDINATE_BAND_M

EPS = np.finfo(float).eps
NORMAL_ROUNDOFF = 64 * EPS
NEAREST_TIE_M = 1e-12
CONE_RESIDUAL_ROUNDOFF = 1e-12


def edge_point(point, a, b):
    edge = b-a
    return a+np.clip((point-a)@edge/(edge@edge), 0, 1)*edge


def triangle_inside(point, triangle):
    a, b, c = triangle
    e, f = b-a, c-a
    uv = np.linalg.solve(np.array([[e@e, e@f], [e@f, f@f]]),
                         np.array([(point-a)@e, (point-a)@f]))
    return min(uv) >= -1e-13 and sum(uv) <= 1+1e-13


def nearest_segment_triangle(s0, s1, triangle):
    """Return all enumerated minima within 1pm; None means unresolved geometry.

    Endpoint/closed-face, segment/closed-edge and face crossing exhaust the
    finite feature cases. Near-parallel ill-conditioned edge solves abstain.
    A tie is not a proof of unique contact location; the caller checks spread.
    """
    a, b, c = triangle
    n = np.cross(b-a, c-a)
    length = np.linalg.norm(n)
    v = s1-s0
    if length == 0 or v@v == 0 or not np.isfinite(np.r_[triangle.ravel(), s0, s1]).all():
        return None
    n = n/length
    pairs = []
    def add(q, p, feature):
        pairs.append((float(np.linalg.norm(q-p)), q, p, feature))
    for q in (s0, s1):
        p = q-((q-a)@n)*n
        if triangle_inside(p, triangle):
            add(q, p, 'endpoint_face')
        for e, f in ((a, b), (b, c), (c, a)):
            add(q, edge_point(q, e, f), 'endpoint_edge')
    for e, f in ((a, b), (b, c), (c, a)):
        u = f-e
        for p in (e, f):
            add(edge_point(p, s0, s1), p, 'edge_endpoint_axis')
        mat = np.array([[v@v, -v@u], [-v@u, u@u]])
        determinant = float(np.linalg.det(mat))
        cross_vu = np.cross(v, u)
        cross2 = float(cross_vu@cross_vu)
        if cross2 != 0 and determinant <= 1e-14*(v@v)*(u@u):
            return None
        if determinant > 1e-14*(v@v)*(u@u):
            st = np.linalg.solve(mat, np.array([(e-s0)@v, -(e-s0)@u]))
            if min(st) >= 0 and max(st) <= 1:
                add(s0+st[0]*v, e+st[1]*u, 'segment_edge_interior')
    denominator = v@n
    if abs(denominator) > 1e-14*np.linalg.norm(v):
        h = ((a-s0)@n)/denominator
        if 0 <= h <= 1:
            q = s0+h*v
            if triangle_inside(q, triangle):
                add(q, q, 'crossing')
    best = min(row[0] for row in pairs)
    return [row for row in pairs if abs(row[0]-best) <= NEAREST_TIE_M]


def near_source_faces(mesh, point):
    """Closed finite triangles within the unchanged native coordinate band."""
    indices = mesh.face_candidates(point, NATIVE_COORDINATE_BAND_M)
    if not len(indices):
        return indices
    a, b, c = (mesh.triangles[indices, i] for i in range(3))
    normals = mesh.normals[indices]
    planes = np.einsum('ij,ij->i', point-a, normals)
    relative = point-planes[:, None]*normals-a
    u = np.einsum('ij,ij->i', np.cross(relative, mesh.edge_b[indices]), normals)/mesh.double_areas[indices]
    v = np.einsum('ij,ij->i', np.cross(mesh.edge_a[indices], relative), normals)/mesh.double_areas[indices]
    inside = (u >= 0) & (v >= 0) & (u+v <= 1)
    distances2 = np.where(inside, planes*planes, np.inf)
    for first, second in ((a, b), (b, c), (c, a)):
        edge = second-first
        t = np.clip(np.einsum('ij,ij->i', point-first, edge)/np.einsum('ij,ij->i', edge, edge), 0, 1)
        delta = point-first-t[:, None]*edge
        distances2 = np.minimum(distances2, np.einsum('ij,ij->i', delta, delta))
    return indices[distances2 <= NATIVE_COORDINATE_BAND_M**2]


def normal_cone(normals, direction):
    """Enumerate nonnegative active sets of at most three generators in R3."""
    best = (1., [], [])
    for count in range(1, min(3, len(normals))+1):
        for ids in itertools.combinations(range(len(normals)), count):
            matrix = normals[list(ids)].T
            coefficients = np.linalg.lstsq(matrix, direction, rcond=None)[0]
            if np.min(coefficients) < -1e-13:
                continue
            coefficients = np.maximum(coefficients, 0.)
            residual = float(np.linalg.norm(matrix@coefficients-direction))
            if residual < best[0]:
                best = residual, list(ids), coefficients.tolist()
    return {'residual': best[0], 'generator_indices': best[1],
            'nonnegative_coefficients': best[2], 'in_cone_roundoff': best[0] <= CONE_RESIDUAL_ROUNDOFF}


class SourcePatches:
    """Immutable source topology cache; no owner pose is retained here."""
    def __init__(self, mesh):
        self.mesh = mesh
        self.location_guard = 64*EPS*max(float(np.max(np.abs(mesh.vertices))), 1.)
        self.edges = {}
        self.face_edges = []
        for face, vertices in enumerate(mesh.faces):
            edges = [tuple(sorted((int(vertices[a]), int(vertices[b]))))
                     for a, b in ((0, 1), (1, 2), (2, 0))]
            self.face_edges.append(edges)
            for edge in edges:
                self.edges.setdefault(edge, []).append(face)
        self.edge_order = {edge: index for index, edge in enumerate(self.edges)}
        self.adjacency = [[] for _ in mesh.faces]
        for faces in self.edges.values():
            assert len(faces) == 2
            a, b = faces
            self.adjacency[a].append(b)
            self.adjacency[b].append(a)
        self.face_patches = {}
        self.patches = {}

    def patch(self, seed):
        if seed in self.face_patches:
            return self.face_patches[seed]
        mesh = self.mesh
        anchor, normal = mesh.triangles[seed, 0], mesh.normals[seed]
        accepted, visited, pending = set(), set(), [seed]
        while pending:
            face = pending.pop()
            if face in visited:
                continue
            visited.add(face)
            if np.linalg.norm(mesh.normals[face]-normal) > NORMAL_ROUNDOFF:
                continue
            if np.max(np.abs((mesh.triangles[face]-anchor)@normal)) > self.location_guard:
                continue
            accepted.add(face)
            pending.extend(self.adjacency[face])
        # Roundoff-based coplanarity must not create inconsistent overlapping patches.
        if any(face in self.face_patches for face in accepted):
            return None
        key = min(accepted)
        # A patch boundary must touch one of its accepted faces. Inspect
        # those edges only, preserving the original mesh-wide edge order.
        incident_edges = {edge for face in accepted for edge in self.face_edges[face]}
        boundary = sorted((edge for edge in incident_edges
                           if sum(face in accepted for face in self.edges[edge]) == 1),
                          key=self.edge_order.__getitem__)
        self.patches[key] = {'faces': sorted(accepted), 'normal': normal, 'boundary': boundary}
        for face in accepted:
            self.face_patches[face] = key
        return key

    def matched_patch(self, point):
        near = near_source_faces(self.mesh, point)
        if not len(near):
            return None, 'no_matched_face', []
        keys = [self.patch(int(face)) for face in near]
        if None in keys or len(set(keys)) != 1:
            return None, 'different_or_unresolved_patch', near.tolist()
        return keys[0], None, near.tolist()

    def boundary_support(self, point, direction, patch):
        mesh = self.mesh
        hits = [edge for edge in patch['boundary']
                if np.linalg.norm(point-edge_point(point, *mesh.vertices[list(edge)])) <= self.location_guard]
        if not hits:
            if np.linalg.norm(direction-patch['normal']) > CONE_RESIDUAL_ROUNDOFF:
                return None, 'interior_normal_not_source_normal'
            return {'classification': 'relative_patch_interior'}, None
        vertices = [vertex for edge in hits for vertex in edge
                    if np.linalg.norm(point-mesh.vertices[vertex]) <= self.location_guard]
        incident = sorted(set(face for edge in hits for face in self.edges[edge])
                          | set(face for vertex in vertices for face in mesh.incident[vertex]))
        delta = mesh.vertices[np.unique(mesh.faces[incident])]-point
        guard = 64*EPS*float(np.max(np.linalg.norm(delta, axis=1)))
        outside = np.max(delta@mesh.normals[incident].T, axis=0)
        if np.max(outside) > guard:
            return None, 'nonconvex_source_boundary'
        unique = []
        for face in incident:
            if not any(np.linalg.norm(mesh.normals[face]-other) <= NORMAL_ROUNDOFF for other in unique):
                unique.append(mesh.normals[face])
        cone = normal_cone(np.array(unique), direction)
        if not cone['in_cone_roundoff']:
            return None, 'normal_outside_source_cone'
        return {'classification': 'true_patch_boundary', 'boundary_edges': [list(edge) for edge in hits],
                'incident_faces': incident, 'incident_normals': mesh.normals[incident].tolist(),
                'max_outside_each_plane_m': outside.tolist(), 'support_roundoff_guard_m': guard,
                'normal_cone': cone}, None

    def nearest(self, ends, radius, key):
        patch = self.patches[key]
        solutions = []
        for face in patch['faces']:
            rows = nearest_segment_triangle(*ends, self.mesh.triangles[face])
            if rows is None:
                return None, 'unresolved_finite_feature_solve'
            solutions.extend((*row, face) for row in rows)
        best = min(row[0] for row in solutions)
        tied = [row for row in solutions if abs(row[0]-best) <= NEAREST_TIE_M]
        d, q, p, feature, face = min(tied, key=lambda row: row[0])
        if not math.isfinite(d) or d <= self.location_guard:
            return None, 'zero_or_unknown_axis_distance'
        if any(max(np.linalg.norm(a-q), np.linalg.norm(b-p)) > self.location_guard for _, a, b, _, _ in tied):
            return None, 'nonunique_contact_location'
        direction = (q-p)/d
        if any(np.linalg.norm((a-b)/dd-direction) > CONE_RESIDUAL_ROUNDOFF for dd, a, b, _, _ in tied):
            return None, 'nonunique_contact_normal'
        if direction@patch['normal'] <= 0:
            return None, 'back_facing'
        evidence, reason = self.boundary_support(p, direction, patch)
        if reason:
            return None, reason
        return {'point_mesh_local_m': p, 'point_capsule_local_m': q-radius*direction,
                'normal_mesh_to_capsule_local': direction, 'gap_m': d-radius,
                'selected_finite_face': face, 'feature': feature, 'source_evidence': evidence}, None


def eligible_capsule(record, created):
    supplements = record.get('source_analytic_material', [])
    if len(supplements) != 1 or len(created) != 1 or supplements != created:
        return None
    cap = supplements[0]
    proof = cap.get('proof', {})
    if (cap.get('type') != 'capsule' or len(record.get('source_solid_ids', [])) != 1
            or cap.get('source_solid_id') not in record['source_solid_ids']
            or proof.get('cut_is_done') is not True or proof.get('missing_volume_mm3') != 0.
            or proof.get('original_solid_modified') is not False
            or proof.get('method') != 'OCCT non-destructive Cut(complete_source_derived_cylinder, original_solid)'):
        return None
    radius, length = cap['radius_m'], cap['straight_length_m']
    axis, center = np.array(cap['axis_source']), np.array(cap['center_local_m'])
    if (not np.isfinite(np.r_[axis, center, radius, length]).all() or radius <= 0 or length <= 0
            or abs(np.linalg.norm(axis)-1.) > NORMAL_ROUNDOFF):
        return None
    return {'radius': radius, 'ends': np.array([center-length*axis/2, center+length*axis/2])}


def build_source_capsule_contacts(c, system, source_filter, records, previous_callback, *, created_capsules_by_body):
    """Wrap the previous callback; caller must use returned begin_step each step.

    created_capsules_by_body maps body IDs to source supplements actually added
    as capsule shapes. Only one unambiguous source capsule per owner is supported.
    """
    sources = source_filter.source_meshes
    capsules = {key: cap for key, record in records.items()
                if (cap := eligible_capsule(record, created_capsules_by_body.get(key, []))) is not None}
    ce = c.ChCollisionShape.CONNECTEDTRIANGLE if hasattr(c.ChCollisionShape, 'CONNECTEDTRIANGLE') else c.ChCollisionShape.MESHTRIANGLE

    class SourceCapsuleContacts(c.NarrowphaseCallback):
        def __init__(self):
            super().__init__()
            self.topology = {}
            self.step_queries = {}
            self.counts = Counter()
            self.changed_samples = []

        def begin_step(self):
            previous_callback.begin_step()
            self.step_queries.clear()

        def unchanged(self, reason):
            self.counts[reason] += 1
            return None

        def reconstruct(self, info):
            self.counts['candidates'] += 1
            types = [info.shapeA.GetType(), info.shapeB.GetType()]
            if types == [ce, c.ChCollisionShape.CAPSULE]:
                mesh_side = 0
            elif types == [c.ChCollisionShape.CAPSULE, ce]:
                mesh_side = 1
            else:
                return self.unchanged('outside_shape_scope')
            keys = [str(model.GetContactable().GetPhysicsItem().GetName()) for model in (info.modelA, info.modelB)]
            mesh_key, capsule_key = keys[mesh_side], keys[1-mesh_side]
            if (capsule_key not in capsules or mesh_key not in sources or capsule_key not in sources
                    or not sources[mesh_key].valid_single_boundary or not sources[capsule_key].valid_single_boundary):
                return self.unchanged('unknown_or_nonunique_source_capsule')
            if mesh_key not in self.topology:
                self.topology[mesh_key] = SourcePatches(sources[mesh_key])
            topology = self.topology[mesh_key]
            rotation, position = source_filter.owner_pose(mesh_key)
            capsule_rotation, capsule_position = source_filter.owner_pose(capsule_key)
            points = [np.array([value.x, value.y, value.z]) for value in (info.vpA, info.vpB)]
            patch, reason, near = topology.matched_patch((points[mesh_side]-position)@rotation)
            if reason:
                return self.unchanged(reason)
            query = mesh_key, capsule_key, patch
            if query not in self.step_queries:
                cap = capsules[capsule_key]
                ends = (cap['ends']@capsule_rotation.T+capsule_position-position)@rotation
                self.step_queries[query] = topology.nearest(ends, cap['radius'], patch)
                self.counts['finite_patch_queries'] += 1
            else:
                self.counts['step_query_cache_hits'] += 1
            result, reason = self.step_queries[query]
            if reason:
                return self.unchanged(reason)
            mesh_point = result['point_mesh_local_m']@rotation.T+position
            capsule_point = result['point_capsule_local_m']@rotation.T+position
            normal = result['normal_mesh_to_capsule_local']@rotation.T
            a, b, normal = ((mesh_point, capsule_point, normal) if mesh_side == 0
                            else (capsule_point, mesh_point, -normal))
            gap = result['gap_m']
            if abs(float((b-a)@normal)-gap) > topology.location_guard:
                return self.unchanged('world_distance_identity_failed')
            before = {'distance_m': float(info.distance), 'point_a_world_m': points[0].tolist(),
                      'point_b_world_m': points[1].tolist(), 'normal_A_to_B_world': [info.vN.x, info.vN.y, info.vN.z]}
            after = {'distance_m': gap, 'point_a_world_m': a.tolist(), 'point_b_world_m': b.tolist(), 'normal_A_to_B_world': normal.tolist()}
            if before == after:
                return self.unchanged('already_source_consistent')
            info.vpA, info.vpB = c.ChVector3d(*a), c.ChVector3d(*b)
            info.vN, info.distance = c.ChVector3d(*normal), gap
            self.counts['reconstructed'] += 1
            return {'time_s': float(system.GetChTime()), 'body_pair': keys, 'mesh_body': mesh_key, 'capsule_body': capsule_key,
                    'matched_source_faces': near, 'source_patch_faces': topology.patches[patch]['faces'],
                    'before': before, 'after': after, 'selected_finite_face': result['selected_finite_face'],
                    'source_evidence': result['source_evidence'], 'source_location_roundoff_m': topology.location_guard,
                    'point_a_migration_m': float(np.linalg.norm(a-points[0])), 'point_b_migration_m': float(np.linalg.norm(b-points[1]))}

        def OnNarrowphase(self, info):
            try:
                change = self.reconstruct(info)
                keep = previous_callback.OnNarrowphase(info)
                if change is not None:
                    self.counts['reconstructed_kept' if keep else 'reconstructed_rejected_by_previous'] += 1
                    if len(self.changed_samples) < 16:
                        change['kept_after_previous'] = bool(keep)
                        self.changed_samples.append(change)
                return keep
            except Exception:
                import traceback
                traceback.print_exc()
                raise

        def metadata(self):
            return {'schema_version': 'source_capsule_finite_patch_contacts_v1',
                    'rule': 'One actually created source-contained capsule, one matched finite original coplanar patch, unique nearest location and outward source-supported normal',
                    'counts': dict(self.counts), 'eligible_capsule_bodies': sorted(capsules),
                    'native_coordinate_band_m': NATIVE_COORDINATE_BAND_M, 'nearest_tie_band_m': NEAREST_TIE_M,
                    'normal_coplanarity_roundoff': NORMAL_ROUNDOFF, 'cone_residual_roundoff': CONE_RESIDUAL_ROUNDOFF,
                    'location_roundoff': '64*float64_epsilon*max(abs(source_mesh_coordinates),1 metre)',
                    'topology_cache': 'Fixed original adjacency and connected coplanar finite patches only',
                    'nearest_cache': 'Current step only; begin_step delegates to previous callback and clears nearest queries',
                    'candidate_deletion_or_deduplication': False, 'geometry_mass_inertia_joints_changed': False,
                    'acceptance_tolerances_changed': False, 'maximum_saved_changes': 16,
                    'changed_contact_samples': deepcopy(self.changed_samples), 'previous_callback_chained': True}

    callback = SourceCapsuleContacts()
    system.GetCollisionSystem().RegisterNarrowphaseCallback(callback)
    return callback
