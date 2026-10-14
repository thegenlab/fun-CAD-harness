import unittest

import numpy as np
import trimesh

from simulation_ready_cad_workflow.validation_pipeline.slim_harness.source_queries import unsigned_distance


def original_distance(mesh, points):
    origin = mesh.bounds.mean(axis=0)
    extent = np.ptp(mesh.vertices, axis=0).max()
    scale = 1000. / extent
    query = trimesh.Trimesh((mesh.vertices - origin) * scale, mesh.faces[mesh.area_faces > 0], process=False)
    return trimesh.proximity.closest_point(query, (points - origin) * scale)[1] / scale


class SourceDistanceTests(unittest.TestCase):
    def compare(self, mesh, points):
        np.testing.assert_allclose(unsigned_distance(mesh, points), original_distance(mesh, points), atol=1e-10, rtol=1e-9)

    def test_through_bore_near_surface_and_distant_points(self):
        mesh = trimesh.creation.annulus(r_min=2, r_max=5, height=6, sections=64)
        random = np.random.default_rng(4).uniform(-8, 8, (320, 3))
        points = np.vstack((random, mesh.vertices[:80], mesh.triangles_center[:80], [[0, 0, 0], [2 + 1e-9, 0, 0], [2 - 1e-9, 0, 0]]))
        self.compare(mesh, points)

    def test_inward_cavity_shell_is_not_filled(self):
        outer = trimesh.creation.box([10, 10, 10])
        inner = trimesh.creation.box([4, 4, 4])
        inner.invert()
        mesh = trimesh.util.concatenate([outer, inner])
        points = np.array([[0, 0, 0], [1.9, 0, 0], [2.1, 0, 0], [4.9, 0, 0], [6, 0, 0], [100, 2, 3.]])
        self.compare(mesh, points)
        self.assertAlmostEqual(unsigned_distance(mesh, [[0, 0, 0]])[0], 2.)

    def test_translated_small_geometry_preserves_normalized_distances(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=.0005)
        mesh.apply_translation([123.4, -25.3, 87.2])
        points = mesh.vertices[::3] + np.array([1e-7, -2e-7, 3e-7])
        self.compare(mesh, points)

    def test_empty_point_set(self):
        self.assertEqual(unsigned_distance(trimesh.creation.box(), []).shape, (0,))

    def test_candidate_budget_flushes_without_losing_far_points(self):
        mesh = trimesh.creation.icosphere(subdivisions=3)
        points = np.random.default_rng(7).uniform(40,60,(256,3))
        self.compare(mesh,points)


if __name__ == "__main__":
    unittest.main()
