"""Conformally split long original triangles without moving their surface."""
from __future__ import annotations

import numpy as np


# Long rail triangles lost an otherwise tangent native capsule/sphere contact.
# Midpoint subdivision recovered it while retaining the exact source facets.
MAX_SOURCE_TRIANGLE_EDGE_M = 0.08


def refine_source_triangles(vertices, faces):
    points = [point.copy() for point in np.asarray(vertices, dtype=float)]
    triangles = np.asarray(faces, dtype=int).tolist()
    while True:
        selected = set()
        for face in triangles:
            edges = [tuple(sorted((a, b))) for a, b in zip(face, face[1:] + face[:1])]
            if any(np.linalg.norm(points[a] - points[b]) > MAX_SOURCE_TRIANGLE_EDGE_M for a, b in edges):
                selected.update(edges)
        if not selected:
            return np.asarray(points), np.asarray(triangles, dtype=int)
        # One midpoint per shared edge also splits each incident neighbour.
        midpoint = {edge: len(points) + index for index, edge in enumerate(sorted(selected))}
        points.extend((points[a] + points[b]) / 2 for a, b in midpoint)
        refined = []
        for face in triangles:
            mids = [midpoint.get(tuple(sorted((face[i], face[(i + 1) % 3])))) for i in range(3)]
            count = sum(value is not None for value in mids)
            if count == 0:
                refined.append(face)
            elif count == 1:
                i = next(i for i in range(3) if mids[i] is not None)
                a, b, c = face[i:] + face[:i]
                m = mids[i]
                refined.extend(((a, m, c), (m, b, c)))
            elif count == 2:
                i = (next(i for i in range(3) if mids[i] is None) + 1) % 3
                a, b, c = face[i:] + face[:i]
                ab, bc = mids[i], mids[(i + 1) % 3]
                refined.extend(((a, ab, c), (ab, bc, c), (ab, b, bc)))
            else:
                a, b, c = face
                ab, bc, ca = mids
                refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        triangles = [list(face) for face in refined]
