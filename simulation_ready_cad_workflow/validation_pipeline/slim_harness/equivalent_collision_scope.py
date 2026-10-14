"""Native contact filtering for the same finite, pair-local geometry scope."""
from __future__ import annotations

import math
import numpy as np

from .validation_scope import validate_validation_scope


def build_equivalent_collision_scope(c, system, bodies, records, scope, *, support_records=None):
    validate_validation_scope(scope, list(records.values()))
    support_records = support_records or {}
    regions_by_pair = {}
    for region in scope["flexible_contact_regions"]:
        if region.get("active_element_id") is not None and region["active_element_id"] not in support_records:
            raise ValueError("Conditional contact region lacks its live shaft-support provider")
        regions_by_pair.setdefault(tuple(sorted(region["body_pair"])), []).append(region)
    if not regions_by_pair:
        return None
    source_positions = {key: row["position_m"] for key, row in records.items()}
    region_constants = {}
    for row in scope["flexible_contact_regions"]:
        origin, axis = tuple(row["axis_origin_source_m"]), tuple(row["axis_direction_source"])
        source = tuple(source_positions[row["region_owner_body"]])
        region_constants[row["id"]] = (origin, axis, *row["axial_interval_m"], row["radius_m"]**2,
            np.asarray(origin), np.asarray(axis), np.asarray(source), max(1., *map(abs, (*origin, *source))))

    class ScopedContactFilter(c.NarrowphaseCallback):
        def __init__(self):
            super().__init__()
            self.excluded_counts = {row["id"]: 0 for row in scope["flexible_contact_regions"]}
            self.last_exclusions = []
            self.owner_poses = {}

        def begin_step(self):
            self.last_exclusions = []
            self.owner_poses = {}

        def owner_pose(self, owner):
            if owner not in self.owner_poses:
                body = bodies[owner]
                p, q = body.GetPos(), body.GetRot()
                position = (float(p.x), float(p.y), float(p.z))
                raw_quaternion = [float(q.e0), float(q.e1), float(q.e2), float(q.e3)]
                quaternion = np.asarray(raw_quaternion, dtype=float)
                quaternion /= np.linalg.norm(quaternion)
                w, x, y, z = quaternion
                rotation = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
                rotation = tuple(tuple(float(v) for v in row) for row in rotation)
                self.owner_poses[owner] = (position, rotation, np.asarray(rotation), np.asarray(position),
                                           max(1., *map(abs, position)))
            return self.owner_poses[owner]

        def contains(self, region, pair, point):
            owner = region["region_owner_body"]
            position, rotation, rotation_array, position_array, pose_scale = self.owner_pose(owner)
            origin, axis, lower, upper, radius2, origin_array, axis_array, source_array, region_scale = region_constants[region["id"]]
            x, y, z = (point[i]-position[i] for i in range(3))
            source = source_positions[owner]
            delta = tuple(((rotation[0][i]*x+rotation[1][i]*y)+rotation[2][i]*z)+source[i]-origin[i]
                          for i in range(3))
            along = (delta[0]*axis[0]+delta[1]*axis[1])+delta[2]*axis[2]
            radial = tuple(delta[i]-along*axis[i] for i in range(3))
            radial2 = (radial[0]*radial[0]+radial[1]*radial[1])+radial[2]*radial[2]
            # Scalar dot products can differ from NumPy by a few ulps. Near
            # an original boundary, use that same original predicate; this
            # band changes the evaluation path, never a radius or interval.
            scale = max(pose_scale, region_scale, abs(point[0]), abs(point[1]), abs(point[2]))
            linear_guard = 32*math.ulp(scale)
            squared_guard = 32*math.ulp(max(1., abs(radial2), abs(radius2)))
            if (min(abs(along-lower), abs(along-upper)) <= linear_guard
                    or abs(radial2-radius2) <= squared_guard):
                # Same NumPy operations as the original boundary predicate,
                # but its unchanged normalized rotation is already cached.
                delta_array = rotation_array.T@(np.asarray(point)-position_array)+source_array-origin_array
                along = float(delta_array@axis_array)
                radial_array = delta_array-along*axis_array
                return lower <= along <= upper and float(radial_array@radial_array) <= radius2
            return lower <= along <= upper and radial2 <= radius2

        def OnNarrowphase(self, info):
            try:
                return self.process_contact(info)
            except Exception:
                # SWIG otherwise replaces the useful callback exception with
                # a generic director error. Preserve the real native failure.
                import traceback
                traceback.print_exc()
                raise

        def process_contact(self, info):
            pair = tuple(sorted((str(info.modelA.GetContactable().GetPhysicsItem().GetName()),
                                 str(info.modelB.GetContactable().GetPhysicsItem().GetName()))))
            regions = regions_by_pair.get(pair)
            if not regions:
                return True
            points = [[float(v.x), float(v.y), float(v.z)] for v in (info.vpA, info.vpB)]
            for region in regions:
                if region.get("active_element_id") is not None and not support_records[region["active_element_id"]]["active"]:
                    continue
                # Both points still have to belong to this SAME finite region.
                if all(self.contains(region, pair, point) for point in points):
                    self.excluded_counts[region["id"]] += 1
                    if len(self.last_exclusions) < 16:
                        self.last_exclusions.append({"region_id": region["id"], "body_pair": list(pair),
                            "native_point_a_abs_m": points[0], "native_point_b_abs_m": points[1],
                            "distance_m": float(info.distance)})
                    return False
            return True

        def metadata(self):
            return {"schema_version": "slim_equivalent_collision_scope_v1",
                    "validation_scope_signature": scope["signature"],
                    "rule": "Exclude only this declared pair with both native surface points inside one finite moving region",
                    "excluded_candidate_counts": dict(self.excluded_counts)}

    callback = ScopedContactFilter()
    system.GetCollisionSystem().RegisterNarrowphaseCallback(callback)
    return callback
