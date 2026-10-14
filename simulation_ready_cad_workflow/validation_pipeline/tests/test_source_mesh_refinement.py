import unittest

import numpy as np
import trimesh

from ..slim_harness.source_mesh_refinement import MAX_SOURCE_TRIANGLE_EDGE_M, refine_source_triangles


class SourceMeshRefinementTests(unittest.TestCase):
    def test_closed_asymmetric_source_keeps_surface_cavity_and_volume(self):
        outer = trimesh.creation.box(extents=[.19, .09, .03])
        inner = trimesh.creation.box(extents=[.13, .045, .015])
        inner.invert()
        source = trimesh.util.concatenate([outer, inner])
        vertices, faces = refine_source_triangles(source.vertices, source.faces)
        result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self.assertTrue(result.is_watertight)
        self.assertTrue(result.is_winding_consistent)
        self.assertAlmostEqual(source.volume, result.volume, places=16)
        self.assertEqual(2, len(result.split(only_watertight=False)))
        self.assertLessEqual(result.edges_unique_length.max(), MAX_SOURCE_TRIANGLE_EDGE_M)
        points = np.vstack((result.vertices, result.triangles_center))
        _, distances, _ = trimesh.proximity.closest_point_naive(source, points)
        self.assertLess(distances.max(), 1e-15)
        _, _, source_faces = trimesh.proximity.closest_point_naive(source, result.triangles_center)
        np.testing.assert_allclose(result.face_normals, source.face_normals[source_faces], atol=1e-14)

    def test_already_short_mesh_is_unchanged(self):
        source = trimesh.creation.box(extents=[.01, .02, .03])
        vertices, faces = refine_source_triangles(source.vertices, source.faces)
        np.testing.assert_array_equal(vertices, source.vertices)
        np.testing.assert_array_equal(faces, source.faces)


if __name__ == '__main__':
    unittest.main()
