"""Isolated proof for construction-only wheel/planar-support discretization.

This module does not change the runtime, friction, audit or a PASS label. Given
an ORIGINAL STEP intersection, it proves that its entire material lies within
both a thin inward extrusion of one actual planar support face and a finite
declared tread annulus. No generic overlap-volume increase is used. Integration
must be explicitly approved as a numerical contact approximation for benchmark
construction; this is not a deformed-rubber or frozen-evaluation certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np


class PlanarContactProofError(ValueError):
    pass


def _unit(value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1., atol=1e-9):
        raise PlanarContactProofError("A finite unit axis is required")
    return vector / np.linalg.norm(vector)


def _bounds(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    # Do not bound an inscribed tessellation; enclose the analytic source.
    BRepBndLib.AddOptimal_s(shape, box, False, True)
    if box.IsVoid():
        return None
    values = np.asarray(box.Get(), dtype=float)
    if not np.isfinite(values).all():
        raise PlanarContactProofError("Unbounded original contact shape")
    return values.reshape(2, 3)


def _volume_m3(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return abs(float(properties.Mass())) * 1e-9


def _transform(shape, matrix, shift_mm):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    transform = gp_Trsf()
    transform.SetValues(*np.column_stack((matrix, shift_mm)).reshape(-1).tolist())
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _body_pose(record, pose):
    from .audit import rotation
    q = np.asarray(pose['rotation_abs_quat_wxyz'], dtype=float)
    p = np.asarray(pose['position_abs_m'], dtype=float)
    if q.shape != (4,) or p.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(p).all() or np.linalg.norm(q) == 0:
        raise PlanarContactProofError("Incomplete or nonfinite actual body pose")
    matrix = rotation(q)
    center = np.asarray(record.get('source_position_m', record['position_m']))
    return matrix, (p - matrix @ center) * 1000.


def _cut(shape, tool):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    operation = BRepAlgoAPI_Cut(shape, tool)
    if not operation.IsDone():
        raise PlanarContactProofError("Original contact-region Boolean cut did not complete")
    return operation.Shape()


def _faces(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    result = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        result.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    return result


def _plane(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.TopAbs import TopAbs_REVERSED
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() != GeomAbs_Plane:
        raise PlanarContactProofError("Declared support face is not an actual STEP plane")
    plane = surface.Plane()
    direction = plane.Axis().Direction()
    normal = np.asarray([direction.X(), direction.Y(), direction.Z()])
    if face.Orientation() == TopAbs_REVERSED:
        normal = -normal
    location = plane.Location()
    return np.asarray([location.X(), location.Y(), location.Z()]), normal


@dataclass
class PlanarSupportContactProof:
    declaration: dict
    records: dict
    support_face: object
    support_band: object
    tread_region: object
    plane_origin_mm: np.ndarray
    plane_normal: np.ndarray
    depth_limit_m: float
    tolerance_components: dict
    roundoff_volume_m3: float = 1e-12
    fast_data: dict | None = None

    @classmethod
    def from_source(cls, declaration, bodies, *, native_penetration_tolerance_m, gravity_m_s2, joints=()):
        from ..intake.step_shape_cache import load_step_shape
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec

        if declaration.get('schema_version') != 'slim_construction_planar_contact_v1' or declaration.get('stage') != 'benchmark_construction':
            raise PlanarContactProofError("Planar contact allowance is construction-only")
        if declaration.get('contact_kind') != 'wheel_tread_on_planar_support':
            raise PlanarContactProofError("Only explicitly declared wheel-tread support contact is covered")
        records = {row['id']: row for row in bodies}
        support, tread = declaration['support_body'], declaration['tread_body']
        if support == tread or set(declaration.get('body_pair', [])) != {support, tread}:
            raise PlanarContactProofError("The declared contact pair is inconsistent")
        if not records[support]['fixed'] or records[tread]['fixed']:
            raise PlanarContactProofError("This proof requires a moving tread and fixed support")
        if any({row['body_a'], row['body_b']} == {support, tread} for row in joints):
            raise PlanarContactProofError("Joint/internal connection interference cannot use a support-contact allowance")
        loaded = {}
        for key in (support, tread):
            record = records[key]
            binding = declaration.get('source_bindings', {}).get(key, {})
            if binding.get('source_sha256') != record['source_sha256'] or binding.get('source_solid_ids') != record['source_solid_ids']:
                raise PlanarContactProofError("Contact declaration is not bound to the current source inventory")
            path = Path(record['source_file'])
            if path.suffix.lower() not in {'.step', '.stp'}:
                raise PlanarContactProofError("Prototype requires original STEP sources on both sides")
            loaded[key] = load_step_shape(path, unit='MM')
            if loaded[key].source_sha256 != record['source_sha256']:
                raise PlanarContactProofError("Source STEP changed after contact declaration")
        solid_id = declaration['support_solid_id']
        if solid_id not in records[support]['source_solid_ids']:
            raise PlanarContactProofError("Support face does not belong to the bound support body")
        solid = loaded[support].solids[int(solid_id.split('_')[-1]) - 1]
        face_index = declaration['support_face_index']
        faces = _faces(solid)
        if isinstance(face_index, bool) or not isinstance(face_index, int) or not 1 <= face_index <= len(faces):
            raise PlanarContactProofError("Invalid original support-face index")
        face = faces[face_index - 1]
        origin, normal = _plane(face)
        gravity = np.asarray(gravity_m_s2, dtype=float)
        if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) == 0:
            raise PlanarContactProofError("A defined gravity direction is required for planar support")
        # The declared plane is a floor, not a conveniently selected wall/slot.
        source_pose = {'position_abs_m': records[support]['position_m'],
                       'rotation_abs_quat_wxyz': records[support]['rotation_wxyz']}
        support_rotation, _ = _body_pose(records[support], source_pose)
        if np.dot(support_rotation @ normal, -gravity / np.linalg.norm(gravity)) < 1 - 1e-8:
            raise PlanarContactProofError("Declared planar face is not an upward support surface")
        _bounds(face)  # Bounded trimmed face; holes and edges remain in extrusion.
        deflection = records[tread].get('fidelity', {}).get('step_linear_deflection_m')
        if not isinstance(deflection, (int, float)) or not math.isfinite(deflection) or deflection < 0:
            raise PlanarContactProofError("Source mesh deflection metadata is required")
        if not math.isfinite(native_penetration_tolerance_m) or native_penetration_tolerance_m <= 0:
            raise PlanarContactProofError("Use the actual positive native penetration threshold")
        components = {'moving_source_nominal_deflection_m': float(deflection),
                      'planar_support_normal_deflection_m': 0.,
                      'native_penetration_tolerance_m': float(native_penetration_tolerance_m),
                      'linear_roundoff_margin_m': 1e-8}
        limit = sum(components.values())
        cap = declaration.get('maximum_depth_m', .00015)
        if not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0 or limit > cap:
            raise PlanarContactProofError("Derived contact allowance exceeds the declared construction depth cap")
        band = BRepPrimAPI_MakePrism(face, gp_Vec(*(-normal * limit * 1000.)), True).Shape()
        region = declaration['tread_region']
        axis = _unit(region['axis_direction_source'])
        center = np.asarray(region['axis_origin_source_m'], dtype=float)
        low, high = map(float, region['axial_interval_m'])
        inner, outer = float(region['inner_radius_m']), float(region['outer_radius_m'])
        if center.shape != (3,) or not np.isfinite(center).all() or not all(math.isfinite(x) for x in (low, high, inner, outer)) or not 0 < inner < outer or not low < high:
            raise PlanarContactProofError("A finite annular tread region excluding the hub is required")
        frame = gp_Ax2(gp_Pnt(*((center + axis * low) * 1000.)), gp_Dir(*axis))
        outer_shape = BRepPrimAPI_MakeCylinder(frame, outer * 1000., (high - low) * 1000.).Shape()
        inner_shape = BRepPrimAPI_MakeCylinder(frame, inner * 1000., (high - low) * 1000.).Shape()
        annulus = _cut(outer_shape, inner_shape)
        result = cls(declaration, records, face, band, annulus, origin, normal, limit, components)
        result._prepare_box_bound_proof(loaded, solid)
        return result

    def _prepare_box_bound_proof(self, loaded, support_solid):
        """Recognize a single real rectangular plate, never an assumed floor."""
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        support, tread = self.declaration['support_body'], self.declaration['tread_body']
        if len(self.records[support]['source_solid_ids']) != 1:
            return
        faces = _faces(support_solid)
        if len(faces) != 6:
            return
        direction = BRepAdaptor_Surface(self.support_face).Plane().XAxis().Direction()
        u = np.asarray([direction.X(), direction.Y(), direction.Z()])
        frame = np.asarray([u, np.cross(self.plane_normal, u), self.plane_normal])
        directions = set()
        try:
            for face in faces:
                normal = frame @ _plane(face)[1]
                axis = int(np.argmax(abs(normal)))
                if abs(normal[axis]) < 1 - 1e-10:
                    return
                directions.add((axis, 1 if normal[axis] > 0 else -1))
        except PlanarContactProofError:
            return
        if directions != {(axis, sign) for axis in range(3) for sign in (-1, 1)}:
            return
        bounds = _bounds(_transform(support_solid, frame, -frame @ self.plane_origin_mm))
        if bounds is None or abs(bounds[1, 2]) > 1e-5 or bounds[0, 2] >= -self.depth_limit_m * 1000.:
            return
        originals, non_tread = [], []
        for solid_id in self.records[tread]['source_solid_ids']:
            part = loaded[tread].solids[int(solid_id.split('_')[-1]) - 1]
            originals.append(part)
            # Precompute hub/web geometry once. Native Booleans get copies so
            # no shared source-shape cache is mutated by their internal state.
            remainder = _cut(_transform(part, np.eye(3), np.zeros(3)), self.tread_region)
            if _bounds(remainder) is not None:
                non_tread.append(remainder)
        self.fast_data = {'frame': frame, 'support_bounds_mm': bounds,
                          'original_parts': originals, 'non_tread_parts': non_tread}

    def prove_body(self, pair, body_poses, *, native_max_penetration_m):
        """Bound ALL possible contact using exact source geometry, no Common.

        The entire moving footprint must be inside the verified plate interior;
        non-tread material must stay above it. Otherwise return unproven so the
        caller retains its original audit or uses the slower intersection proof.
        """
        started = time.monotonic()
        result = {'schema_version': 'slim_construction_planar_contact_proof_v1', 'allowed': False,
                  'body_pair': sorted(pair), 'depth_limit_m': self.depth_limit_m,
                  'tolerance_components': self.tolerance_components,
                  'scope': 'construction-only numerical planar-support contact approximation; exact source bounds, not rubber deformation',
                  'native_contacts_or_friction_modified': False, 'original_geometry_modified': False,
                  'method': 'exact_original_body_bounds_and_verified_finite_box_support'}
        try:
            if sorted(pair) != sorted(self.declaration['body_pair']):
                result['reason'] = 'unrelated_body_pair'
                return result
            if self.fast_data is None:
                result['reason'] = 'unproven_support_is_not_a_single_verified_rectangular_plate'
                return result
            if not math.isfinite(native_max_penetration_m) or not 0 <= native_max_penetration_m <= self.tolerance_components['native_penetration_tolerance_m']:
                result['reason'] = 'native_contact_depth_exceeds_unchanged_runtime_tolerance'
                return result
            support, tread = self.declaration['support_body'], self.declaration['tread_body']
            sr, sp = _body_pose(self.records[support], body_poses[support])
            tr, tp = _body_pose(self.records[tread], body_poses[tread])
            frame = self.fast_data['frame'] @ sr.T
            origin = sr @ self.plane_origin_mm + sp
            local_rotation, shift = frame @ tr, frame @ (tp - origin)
            boxes = [_bounds(_transform(part, local_rotation, shift)) for part in self.fast_data['original_parts']]
            lower = np.min([box[0] for box in boxes], axis=0)
            upper = np.max([box[1] for box in boxes], axis=0)
            depth = max(0., -float(lower[2]) * .001)
            result['maximum_depth_m'] = depth
            if depth > self.depth_limit_m:
                result['reason'] = 'original_depth_exceeds_contact_band'
                return result
            support_bounds = self.fast_data['support_bounds_mm']
            if np.any(lower[:2] <= support_bounds[0, :2] + 1e-5) or np.any(upper[:2] >= support_bounds[1, :2] - 1e-5):
                result['reason'] = 'unproven_moving_footprint_reaches_support_edges'
                return result
            for part in self.fast_data['non_tread_parts']:
                bound = _bounds(_transform(part, local_rotation, shift))
                if bound[0, 2] <= 1e-5:
                    result['reason'] = 'unproven_non_tread_material_reaches_support'
                    return result
            result.update(allowed=True, reason='all_possible_original_overlap_inside_contact_band_and_tread')
            return result
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            result.update(reason='proof_unresolved', error=str(exc))
            return result
        finally:
            result['elapsed_seconds'] = time.monotonic() - started

    def prove(self, pair, overlap_shape_world_mm, body_poses, *, native_max_penetration_m):
        """Check a saved native pose and an already-computed original intersection.

        The Boolean roundoff remainder is the existing audit tolerance, not an
        enlarged volume allowance. A separate analytic depth bound rejects deep
        contacts regardless of their tiny volume.
        """
        started = time.monotonic()
        result = {'schema_version': 'slim_construction_planar_contact_proof_v1', 'allowed': False,
                  'body_pair': sorted(pair), 'depth_limit_m': self.depth_limit_m,
                  'tolerance_components': self.tolerance_components,
                  'scope': 'construction-only numerical planar-support contact approximation; not rubber deformation or full original-rigid nonpenetration',
                  'native_contacts_or_friction_modified': False, 'original_geometry_modified': False}
        try:
            if sorted(pair) != sorted(self.declaration['body_pair']):
                result['reason'] = 'unrelated_body_pair'
                return result
            if not math.isfinite(native_max_penetration_m) or not 0 <= native_max_penetration_m <= self.tolerance_components['native_penetration_tolerance_m']:
                result['reason'] = 'native_contact_depth_exceeds_unchanged_runtime_tolerance'
                return result
            support, tread = self.declaration['support_body'], self.declaration['tread_body']
            sr, sp = _body_pose(self.records[support], body_poses[support])
            tr, tp = _body_pose(self.records[tread], body_poses[tread])
            normal = sr @ self.plane_normal
            origin = sr @ self.plane_origin_mm + sp
            tangent = _unit(np.cross(normal, [1., 0., 0.] if abs(normal[0]) < .9 else [0., 1., 0.]))
            frame = np.asarray([tangent, np.cross(normal, tangent), normal])
            projected = _transform(overlap_shape_world_mm, frame, -frame @ origin)
            bounds = _bounds(projected)
            if bounds is None or _volume_m3(overlap_shape_world_mm) == 0:
                result.update(allowed=True, reason='no_original_material_overlap', maximum_depth_m=0.)
                return result
            depth = max(0., -float(bounds[0, 2]) * .001)
            result.update(maximum_depth_m=depth, original_overlap_m3=_volume_m3(overlap_shape_world_mm))
            if depth > self.depth_limit_m:
                result['reason'] = 'original_depth_exceeds_contact_band'
                return result
            outside_band = _cut(overlap_shape_world_mm, _transform(self.support_band, sr, sp))
            outside_tread = _cut(overlap_shape_world_mm, _transform(self.tread_region, tr, tp))
            band_remainder, tread_remainder = _volume_m3(outside_band), _volume_m3(outside_tread)
            result.update(outside_support_band_m3=band_remainder, outside_tread_region_m3=tread_remainder,
                          original_audit_roundoff_volume_m3=self.roundoff_volume_m3)
            result['allowed'] = max(band_remainder, tread_remainder) <= self.roundoff_volume_m3
            result['reason'] = 'bounded_original_contact_in_declared_tread' if result['allowed'] else 'overlap_outside_declared_contact_region'
            return result
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            result.update(allowed=False, reason='proof_unresolved', error=str(exc))
            return result
        finally:
            result['elapsed_seconds'] = time.monotonic() - started
