"""Advisory check for floating parts in an as-built assembly.

A joint connects two parts in the simulation even when the CAD leaves them
apart, so a design can simulate while its parts visibly hang in the air: a pin
that never reaches its bore, a collar hovering above the shaft. This check looks
at the original geometry in its as-built pose:

* a joint is physically engaged when, along its axis near the anchor, one part
  contains the axis (pin, shaft) while the other surrounds it (bore, eye,
  socket), or when the two parts touch near the anchor;
* every part should reach a fixed part through engaged joints or touching
  parts; a group that does not is floating.

Findings are advice for the author, never a pass/fail result: free test
objects, captive parts and spring-hung parts can legitimately start apart.
"""
from __future__ import annotations

import itertools
from pathlib import Path

SCHEMA = "slim_floating_parts_v1"
AXIS_SAMPLES = 21
ENCLOSING_RAYS = 12
ENCLOSING_HITS = 7


def _tolerance(diagonal: float) -> float:
    """Touching distance for a part of this size: 2 % of it, kept within 0.2–1 mm."""
    return min(max(.02 * diagonal, .0002), .001)


def _load(bodies):
    import numpy as np
    import trimesh
    meshes, fixed = {}, set()
    for body in bodies:
        mesh = trimesh.load(body["mesh_file"], force="mesh", process=False)
        transform = trimesh.transformations.quaternion_matrix(body["rotation_wxyz"])
        transform[:3, 3] = np.asarray(body["position_m"], dtype=float)
        meshes[body["id"]] = mesh.copy().apply_transform(transform)
        if body.get("fixed"):
            fixed.add(body["id"])
    return meshes, fixed


def _world_points(region, body):
    import numpy as np
    import trimesh
    rotation = trimesh.transformations.quaternion_matrix(body["rotation_wxyz"])[:3, :3]
    return [rotation @ np.asarray(point, dtype=float) + np.asarray(body["position_m"], dtype=float)
            for point in (region or {}).get("points") or []]


def _surrounds(mesh, point, axis, reach) -> bool:
    """Most rays across the axis hit this part's material within reach: a bore, eye or socket."""
    import numpy as np
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.arange(ENCLOSING_RAYS) * 2 * np.pi / ENCLOSING_RAYS
    directions = np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)
    locations, rays, _ = mesh.ray.intersects_location(np.repeat(point[None], ENCLOSING_RAYS, axis=0), directions,
                                                     multiple_hits=False)
    hits = {int(ray) for location, ray in zip(locations, rays) if np.linalg.norm(location - point) <= reach}
    return len(hits) >= ENCLOSING_HITS


def _collision_object(mesh):
    import fcl
    import numpy as np
    model = fcl.BVHModel()
    model.beginModel(len(mesh.vertices), len(mesh.faces))
    model.addSubModel(np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int32))
    model.endModel()
    return fcl.CollisionObject(model, fcl.Transform())


def _gap(first, second) -> float:
    """Smallest distance between two parts (0 when they touch or overlap)."""
    import fcl
    return max(0.0, float(fcl.distance(first, second, fcl.DistanceRequest(), fcl.DistanceResult())))


def check_assembly(bodies: list[dict], joints: list[dict]) -> dict:
    """Engagement of every joint and the parts not connected to a fixed part.

    ``bodies``: id, fixed, mesh_file (body-local metres), position_m, rotation_wxyz.
    ``joints``: prepared joints in metres (anchor and axis in the world, region
    points local to their body)."""
    import numpy as np
    meshes, fixed = _load(bodies)
    objects = {key: _collision_object(mesh) for key, mesh in meshes.items()}
    by_id = {body["id"]: body for body in bodies}
    diagonals = {key: float(np.linalg.norm(mesh.extents)) for key, mesh in meshes.items()}
    links, joint_rows, findings = [], [], []
    for joint in joints:
        a, b = joint.get("body_a"), joint.get("body_b")
        if a not in meshes or b not in meshes:
            continue
        anchor = np.asarray(joint["anchor"], dtype=float)
        axis = np.asarray(joint.get("axis") or [0, 0, 1], dtype=float)
        axis /= np.linalg.norm(axis) or 1.0
        regions = [*_world_points(joint.get("region_a"), by_id[a]), *_world_points(joint.get("region_b"), by_id[b])]
        radius = max([float((joint.get(side) or {}).get("radius") or 0) for side in ("region_a", "region_b")] + [.001])
        offsets = [float((point - anchor) @ axis) for point in regions] or [0.0]
        low, high = min(min(offsets), -radius), max(max(offsets), radius)
        samples = [anchor] if joint.get("type") in ("spherical", "ball") else [anchor + t * axis for t in np.linspace(low, high, AXIS_SAMPLES)]
        engaged = 0
        for point in samples:
            inside = {body: bool(meshes[body].contains([point])[0]) for body in (a, b)}
            if inside[a] != inside[b]:
                outer = b if inside[a] else a
                engaged += _surrounds(meshes[outer], point, axis, 1.5 * radius)
        gap = _gap(objects[a], objects[b])
        tolerance = _tolerance(min(diagonals[a], diagonals[b]))
        ok = engaged > 0 or gap <= tolerance
        if ok:
            links.append((a, b))
        joint_rows.append({"joint": joint.get("id"), "body_a": a, "body_b": b, "engaged_samples": engaged,
                           "gap_mm": round(gap * 1000, 2), "physically_engaged": ok})
        if not ok:
            findings.append({"kind": "JOINT_NOT_ENGAGED", "joint": joint.get("id"), "bodies": [a, b],
                             "gap_mm": round(gap * 1000, 2), "message": (
                f"Joint {joint.get('id')} connects {a} and {b} only in the simulation: no pin of one part sits "
                f"in a bore, eye or socket of the other at the anchor, and the parts are {gap * 1000:.2f} mm apart. "
                "Add the material that carries this joint (pin, boss, eye) or move the anchor to it; a larger "
                "joint region radius does not fix it.")})
    for x, y in itertools.combinations(sorted(meshes), 2):
        if (x, y) in links or (y, x) in links:
            continue
        if _gap(objects[x], objects[y]) <= _tolerance(min(diagonals[x], diagonals[y])):
            links.append((x, y))
    neighbours = {key: set() for key in meshes}
    for x, y in links:
        neighbours[x].add(y)
        neighbours[y].add(x)
    components, seen = [], set()
    for start in sorted(meshes):
        if start in seen:
            continue
        group, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node not in group:
                group.add(node)
                stack.extend(neighbours[node] - group)
        seen |= group
        components.append(sorted(group))
    for group in components:
        if fixed and not set(group) & fixed:
            supported = [key for key in meshes if key not in group]
            nearest = min(((_gap(objects[part], objects[other]), part, other) for part in group for other in supported),
                          default=None)
            findings.append({"kind": "FLOATING_PART", "bodies": group,
                             "nearest": nearest and {"body": nearest[2], "gap_mm": round(nearest[0] * 1000, 2)},
                             "message": (
                f"{', '.join(group)} {'is' if len(group) == 1 else 'are'} not physically attached to the fixed parts: "
                "it touches no supported part and no engaged joint holds it"
                + (f" (nearest supported part {nearest[2]}, {nearest[0] * 1000:.2f} mm away)" if nearest else "")
                + ". Unless it is meant to start free (e.g. a dropped test object), connect it with real geometry.")})
    return {"schema_version": SCHEMA, "advisory": True, "joints": joint_rows, "components": components,
            "fixed_bodies": sorted(fixed), "findings": findings}


def advisory_report(bodies: list[dict], joints: list[dict], output) -> dict | None:
    """Run the check when this distribution gives the advice; save it and never fail the caller."""
    from ..mechanism.io import write_json_atomic
    from . import execution_profile
    if not execution_profile.FLOATING_PART_ADVICE:
        return None
    try:
        report = check_assembly(bodies, joints)
    except Exception as exc:  # advice only: a geometry the check cannot read must not stop a run
        report = {"schema_version": SCHEMA, "advisory": True, "findings": [], "error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(output, report)
    return report


def check_prepared(prepared_file) -> dict:
    """The check for a harness-prepared model (prepared.json)."""
    import json
    prepared = json.loads(Path(prepared_file).read_text(encoding="utf-8"))
    return check_assembly(prepared["bodies"], prepared.get("joints") or [])
