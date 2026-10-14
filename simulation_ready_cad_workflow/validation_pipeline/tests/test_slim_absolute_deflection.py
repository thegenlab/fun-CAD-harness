"""Analytic curved-source checks for the stated absolute STEP mesh setting."""
from __future__ import annotations

import json
import unittest

import numpy as np
import trimesh

from ..slim_harness.geometry import (
    STEP_ANGULAR_TOLERANCE_RAD, STEP_DEFLECTION_MM, _closed_mesh, _tessellate_step_mm,
)


def cylinder_chord_deflection_mm(vertices, faces, radius):
    """Exact cylinder radial deficit at lateral-facet edge midpoints."""
    vertices = np.asarray(vertices, dtype=float)
    triangles = vertices[np.asarray(faces, dtype=int)]
    radial = np.linalg.norm(triangles[:, :, :2], axis=2)
    side = (np.ptp(triangles[:, :, 2], axis=1) > 1e-8) & np.all(np.abs(radial - radius) < 1e-7, axis=1)
    triangles = triangles[side]
    if not len(triangles):
        raise AssertionError("Fixture has no original cylinder-side facets")
    midpoint = np.concatenate([(triangles[:, a] + triangles[:, b]) / 2 for a, b in ((0, 1), (1, 2), (2, 0))])
    return float(np.max(radius - np.linalg.norm(midpoint[:, :2], axis=1)))


class SlimAbsoluteDeflectionTests(unittest.TestCase):
    def test_relative_wrapper_breaks_absolute_claim_and_native_absolute_restores_it(self):
        import xml.parsers.expat
        from build123d import Solid
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps

        observations = []
        for radius in (10.0, 250.0):
            raw = BRepPrimAPI_MakeCylinder(radius, 100.0).Shape()
            before = GProp_GProps()
            BRepGProp.VolumeProperties_s(raw, before)
            old_vertices, old_faces = Solid(raw).tessellate(STEP_DEFLECTION_MM, STEP_ANGULAR_TOLERANCE_RAD)
            old_vertices = np.asarray([tuple(vertex) for vertex in old_vertices])
            old_error = cylinder_chord_deflection_mm(old_vertices, old_faces, radius)
            # Intentionally remesh the same BRep after the coarse relative mesh.
            vertices, faces = _tessellate_step_mm(raw)
            new_error = cylinder_chord_deflection_mm(vertices, faces, radius)
            after = GProp_GProps()
            BRepGProp.VolumeProperties_s(raw, after)
            self.assertAlmostEqual(before.Mass(), after.Mass(), places=8)
            self.assertLessEqual(new_error, STEP_DEFLECTION_MM + 1e-8)
            mesh = _closed_mesh(trimesh.Trimesh(vertices=vertices * .001, faces=faces, process=False), "analytic cylinder")
            self.assertTrue(mesh.is_watertight and mesh.is_winding_consistent)
            observations.append({"radius_mm": radius, "requested_absolute_deflection_mm": STEP_DEFLECTION_MM,
                                 "old_relative_wrapper_deflection_mm": old_error,
                                 "new_absolute_deflection_mm": new_error,
                                 "old_triangle_count": len(old_faces), "new_triangle_count": len(faces),
                                 "brep_volume_mm3_before": before.Mass(), "brep_volume_mm3_after": after.Mass()})
        print(json.dumps({"absolute_deflection_counterexample": observations}), flush=True)
        self.assertGreater(observations[1]["old_relative_wrapper_deflection_mm"], STEP_DEFLECTION_MM)

    def test_located_curved_cavity_keeps_world_placement_and_face_orientation(self):
        import xml.parsers.expat
        from build123d import Cylinder
        from OCP.TopLoc import TopLoc_Location
        from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

        ring = Cylinder(10, 30) - Cylinder(5, 32)
        transform = gp_Trsf()
        transform.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), np.pi / 2)
        transform.SetTranslationPart(gp_Vec(40, -20, 30))
        located = ring.wrapped.Located(TopLoc_Location(transform))
        vertices, faces = _tessellate_step_mm(located)
        mesh = _closed_mesh(trimesh.Trimesh(vertices=vertices * .001, faces=faces, process=False), "located cavity")
        expected_bounds_mm = np.asarray([[25, -30, 20], [55, -10, 40]])
        np.testing.assert_allclose(mesh.bounds * 1000, expected_bounds_mm, atol=STEP_DEFLECTION_MM, rtol=0)
        self.assertTrue(mesh.is_watertight and mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0)
        self.assertLess(mesh.volume, np.pi * .010**2 * .030)


if __name__ == "__main__":
    unittest.main()
