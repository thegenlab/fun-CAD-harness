"""Mechanical-law checks for finite shaft supports; native smoke is separate."""
from copy import deepcopy
from types import SimpleNamespace
import math
import unittest
import numpy as np

from .test_slim_equivalent_elasticity import Body, Vector, rotation, vec
from ..slim_harness.shaft_supports import ShaftSupport, compression_response, validate_support


class SupportBody(Body):
    def __init__(self, fixed=False):
        super().__init__([0, 0, 0], np.eye(3))
        self.fixed = fixed
        self.mass = .14
        self.inertia = np.asarray([9e-5, 9e-5, 1.6e-5])
    def IsFixed(self): return self.fixed
    def GetMass(self): return self.mass
    def GetInertiaXX(self): return Vector(*self.inertia)
    def GetInertiaXY(self): return Vector()


def support_spec(*, friction=0., points=8):
    spec = {"id": "support", "kind": "shaft_support", "body_a": "shaft", "body_b": "frame",
        "source_body_ids": ["frame"], "parameter_basis": {"calibration": "not_calibrated"},
        "axis_abs": [0, 0, 1], "anchor_abs_m": [0, 0, 0],
        "axis_a_body": [0, 0, 1], "axis_b_body": [0, 0, 1],
        "anchor_a_body_m": [0, 0, 0], "anchor_b_body_m": [0, 0, 0],
        "friction_regularization_m_s": 1e-5,
        "release": {"radial_offset_m": .01, "axial_offset_m": .01, "tilt_rad": .2},
        "bearings": [], "axial_stops": []}
    for index, z in enumerate([-.04, .04]):
        spec["bearings"].append({"id": f"bearing_{index}", "center_abs_m": [0, 0, z],
            "point_a_body_m": [0, 0, z], "point_b_body_m": [0, 0, z],
            "clearance_m": 5e-5, "shaft_radius_m": .005,
            "stiffness_n_m": 1e4, "damping_n_s_m": 0., "friction": friction})
    for side in [-1, 1]:
        for index in range(points):
            angle = 2*math.pi*index/points
            x, y = .00799*math.cos(angle), .00799*math.sin(angle)
            spec["axial_stops"].append({"id": f"stop_{side}_{index}",
                "point_a_abs_m": [x, y, side*.05005], "point_a_body_m": [x, y, side*.05005],
                "point_b_body_m": [x, y, side*.05], "normal_abs": [0, 0, side],
                "normal_b_body": [0, 0, side], "gap_m": 5e-5,
                "stiffness_n_m": 1e4/points, "damping_n_s_m": 0., "friction": friction})
    return spec


def make_support(spec=None):
    bodies = {"shaft": SupportBody(), "frame": SupportBody(True)}
    handle = ShaftSupport(SimpleNamespace(ChVector3d=Vector), bodies, spec or support_spec(), {key: 0 for key in bodies})
    return handle, bodies


class ShaftSupportTests(unittest.TestCase):
    def test_open_clearance_does_not_hold_or_drag_a_spinning_shaft(self):
        handle, bodies = make_support(support_spec(friction=.3))
        shaft = bodies["shaft"]
        shaft.position = np.asarray([2e-5, 1e-5, 4e-5])
        shaft.matrix = rotation([0, 0, 1], 17.3*math.pi)
        shaft.omega = np.asarray([0., 0., 200.])
        handle.before_step(5e-5, 0.)
        self.assertTrue(handle.record["active"])
        self.assertEqual([], handle.applied_records)
        np.testing.assert_array_equal(shaft.force, [0., 0., 0.])
        np.testing.assert_array_equal(shaft.torque, [0., 0., 0.])

    def test_radial_support_is_finite_and_opposed_stops_retain_axial_travel(self):
        handle, bodies = make_support()
        shaft = bodies["shaft"]
        shaft.position = np.asarray([125e-6, 0., 0.])
        handle.before_step(5e-5, 0.)
        self.assertAlmostEqual(-1.5, shaft.force[0], places=10)
        self.assertAlmostEqual(0., shaft.force[2], places=10)
        self.assertAlmostEqual(0., shaft.torque[2], places=10)
        for side in [-1, 1]:
            current, current_bodies = make_support()
            current_bodies["shaft"].position[2] = side*75e-6
            current.before_step(5e-5, 0.)
            self.assertAlmostEqual(-side*.25, current_bodies["shaft"].force[2], places=9)

    def test_elastic_wrenches_are_the_full_spatial_potential_gradient(self):
        handle, bodies = make_support()
        bodies["shaft"].position = np.asarray([100e-6, 20e-6, 70e-6])
        bodies["shaft"].matrix = rotation([1., 2., 0.], .002)
        bodies["frame"].matrix = rotation([0., 1., 1.], .0002)
        handle.before_step(5e-5, 0.)
        for body in bodies.values():
            position, matrix = body.position.copy(), body.matrix.copy()
            for axis in range(3):
                direction = np.eye(3)[axis]
                h = 1e-8
                body.position = position+h*direction
                plus = handle._evaluate(False)["potential_energy_j"]
                body.position = position-h*direction
                minus = handle._evaluate(False)["potential_energy_j"]
                body.position = position
                self.assertAlmostEqual(-(plus-minus)/(2*h), body.force[axis], delta=2e-5)
                h = 1e-6
                body.matrix = rotation(direction, h)@matrix
                plus = handle._evaluate(False)["potential_energy_j"]
                body.matrix = rotation(direction, -h)@matrix
                minus = handle._evaluate(False)["potential_energy_j"]
                body.matrix = matrix
                self.assertAlmostEqual(-(plus-minus)/(2*h), body.torque[axis], delta=2e-6)
        np.testing.assert_allclose(sum((body.force for body in bodies.values()), np.zeros(3)), 0., atol=1e-12)
        world_torque = sum((body.torque+np.cross(body.position, body.force) for body in bodies.values()), np.zeros(3))
        np.testing.assert_allclose(world_torque, 0., atol=1e-12)

    def test_many_stop_points_share_one_friction_impulse_budget(self):
        handle, bodies = make_support(support_spec(friction=.3, points=32))
        shaft = bodies["shaft"]
        shaft.position[2] = 75e-6
        shaft.omega[2] = 1e-4
        dt = 1.
        handle.before_step(dt, 0.)
        force, torque = np.zeros(3), np.zeros(3)
        for row in handle.applied_records:
            if row["body_id"] == "shaft" and ":friction:" in row["id"]:
                force += row["force_abs_n"]
                torque += np.cross(np.asarray(row["point_abs_m"])-shaft.position, row["force_abs_n"])
        after_v = shaft.velocity+dt*force/shaft.mass
        after_w = shaft.omega+dt*torque/shaft.inertia
        before_energy = .5*float(shaft.inertia@(shaft.omega**2))
        after_energy = .5*shaft.mass*float(after_v@after_v)+.5*float(shaft.inertia@(after_w**2))
        self.assertLess(handle.applied_friction_scale, 1.)
        self.assertGreaterEqual(after_w[2], -1e-12)
        self.assertLessEqual(after_energy, before_energy+1e-16)

    def test_release_withdraws_all_loads_and_is_irreversible(self):
        handle, bodies = make_support(support_spec(friction=.3))
        bodies["shaft"].position[0] = .02
        handle.before_step(5e-5, 1.)
        self.assertFalse(handle.record["active"])
        self.assertEqual([], handle.applied_records)
        self.assertGreater(handle.release_dissipation_j, 0.)
        handle.after_step()
        bodies["shaft"].position[:] = 0.
        handle.before_step(5e-5, 1.00005)
        self.assertFalse(handle.record["active"])
        self.assertEqual([], handle.applied_records)
        self.assertEqual(0., handle.sample()["potential_energy_j"])

    def test_final_integrated_pose_releases_without_erasing_completed_step_forces(self):
        handle, bodies = make_support()
        bodies["shaft"].position[0] = 125e-6
        handle.before_step(5e-5, 2.)
        completed_forces = deepcopy(handle.applied_records)
        self.assertTrue(completed_forces)
        bodies["shaft"].position[0] = .02
        handle.after_step()
        state = handle.sample()
        self.assertFalse(state["active"])
        self.assertFalse(state["within_support_range"])
        self.assertEqual(completed_forces, handle.applied_records)
        self.assertEqual(2.00005, handle.events[-1]["time_s"])

    def test_unilateral_damping_does_not_attract_or_supply_power(self):
        for penetration in [-1e-4, 0., 1e-4]:
            for rate in [-100., -.1, 0., .1, 100.]:
                normal, _, loss = compression_response(penetration, rate, 1e5, 20.)
                self.assertGreaterEqual(normal, 0.)
                self.assertGreaterEqual(loss, 0.)
                if penetration <= 0:
                    self.assertEqual(normal, 0.)
                else:
                    self.assertAlmostEqual(loss, (normal-1e5*penetration)*rate)

    def test_incomplete_support_cannot_claim_both_tilt_and_axial_retention(self):
        spec = support_spec()
        validate_support(spec)
        broken = deepcopy(spec)
        broken["bearings"][1]["center_abs_m"] = broken["bearings"][0]["center_abs_m"]
        with self.assertRaises(ValueError): validate_support(broken)
        broken = deepcopy(spec)
        broken["axial_stops"] = [row for row in broken["axial_stops"] if row["normal_abs"][2] > 0]
        with self.assertRaises(ValueError): validate_support(broken)


if __name__ == "__main__":
    unittest.main()
