"""Analytical potential/gradient and representation checks; no native physics execution."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import numpy as np

from ..slim_harness.equivalent_elasticity import build_equivalent_elasticity, inertia_partition, scalar_response


def base(kind):
    return {"id": "test", "kind": kind, "body_a": "a", "body_b": "b", "source_body_ids": ["a"], "parameter_basis": {"calibration": "not_calibrated"},
            "point_a_body_m": [.1, .02, .03], "point_b_body_m": [-.04, .01, .02], "axis_a_body": [0., 0., 1.],
            "stiffness_n_m": 20., "rest_displacement_m": .01, "rest_length_m": .01, "damping_n_s_m": 0.,
            "response": "bilateral", "working_range_m": [-1., 1.], "anchor_abs_m": [0., 0., 0.], "axis_abs": [0., 0., 1.],
            "stiffness_nm_rad": .3, "rest_angle_rad": 0., "damping_nm_s_rad": 0., "working_range_rad": [-100., 100.],
            "preload_displacement_m": .002, "profile_reference_height_m": 0., "normal_damping_n_s_m": 0.,
            "profile_knots": [{"coordinate_m": -1., "height_m": 0.}, {"coordinate_m": 0., "height_m": .01}, {"coordinate_m": 1., "height_m": 0.}]}


class Vector:
    def __init__(self, x=0., y=0., z=0.): self.x, self.y, self.z = x, y, z


def vec(value): return np.asarray([value.x, value.y, value.z])


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float); axis /= np.linalg.norm(axis)
    x,y,z = axis
    skew = np.asarray([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)*math.cos(angle)+(1-math.cos(angle))*np.outer(axis,axis)+math.sin(angle)*skew


class Body:
    def __init__(self, position, matrix):
        self.position, self.matrix = np.asarray(position, dtype=float), matrix
        self.velocity, self.omega = np.zeros(3), np.zeros(3)
        self.force, self.torque = np.zeros(3), np.zeros(3)
    def AddAccumulator(self): return 0
    def EmptyAccumulator(self, _): self.force[:]=0.; self.torque[:]=0.
    def AccumulateForce(self, _, force, point, local):
        assert not local
        self.force += vec(force); self.torque += np.cross(vec(point)-self.position, vec(force))
    def AccumulateTorque(self, _, torque, local): self.torque += vec(torque)
    def GetPos(self): return Vector(*self.position)
    def GetPosDt(self): return Vector(*self.velocity)
    def GetAngVelParent(self): return Vector(*self.omega)
    def TransformPointLocalToParent(self, point): return Vector(*(self.position+self.matrix@vec(point)))
    def TransformDirectionLocalToParent(self, direction): return Vector(*(self.matrix@vec(direction)))
    def TransformDirectionParentToLocal(self, direction): return Vector(*(self.matrix.T@vec(direction)))


class System:
    def __init__(self): self.time=0.
    def GetChTime(self): return self.time


class EquivalentElasticityTests(unittest.TestCase):
    def test_elastic_efforts_are_potential_gradients(self):
        for kind,q in [("torsional_spring",.7),("linear_spring",.017),("point_axis_spring",.017),("profile_detent",.3)]:
            spec=base(kind);step=1e-7
            derivative=(scalar_response(spec,q+step,0)["potential_energy_j"]-scalar_response(spec,q-step,0)["potential_energy_j"])/(2*step)
            self.assertAlmostEqual(-derivative,scalar_response(spec,q,0)["generalized_force"],delta=1e-8)
        spec=base("point_axis_spring");spec["response"]="compression_only"
        self.assertEqual(0.,scalar_response(spec,.02,-1)["generalized_force"])
        self.assertGreater(scalar_response(spec,0.,0)["generalized_force"],0.)

    def test_damping_never_supplies_positive_power_or_compression_attraction(self):
        for kind in ("torsional_spring","linear_spring","point_axis_spring","profile_detent"):
            for rate in (-2.,-.1,.1,2.):
                spec=base(kind);spec.update(damping_nm_s_rad=.4,damping_n_s_m=.4,normal_damping_n_s_m=.4)
                if kind=="point_axis_spring": spec["response"]="compression_only"
                q=0. if kind=="point_axis_spring" else .3
                zero=deepcopy(spec);zero.update(damping_nm_s_rad=0.,damping_n_s_m=0.,normal_damping_n_s_m=0.)
                damped=scalar_response(spec,q,rate);elastic=scalar_response(zero,q,rate)
                power=(damped["generalized_force"]-elastic["generalized_force"])*rate
                self.assertLessEqual(power,1e-12)
                self.assertAlmostEqual(-power,damped["dissipation_rate_w"],delta=1e-12)
                if kind=="point_axis_spring":self.assertGreaterEqual(damped["generalized_force"],0.)

    def test_full_point_axis_body_wrenches_match_spatial_energy_gradient(self):
        c=SimpleNamespace(ChVector3d=Vector);system=System()
        bodies={"a":Body([.2,.1,0],rotation([0,1,0],.4)),"b":Body([.25,.18,.1],rotation([0,0,1],.2))}
        spec=base("point_axis_spring")
        declaration={"schema_version":"slim_equivalent_elasticity_v1","units":"m","elastic_reference_body_ids":[],"flexible_contact_regions":[],"elements":[spec]}
        handle=build_equivalent_elasticity(c,system,bodies,{},declaration)
        handle.before_step(.001)
        def energy():
            state=handle._state(handle.entries[0]);return scalar_response(spec,state["coordinate"],0)["potential_energy_j"]
        delta=1e-7
        for body in bodies.values():
            for axis in range(3):
                original=body.position.copy();body.position[axis]+=delta;positive=energy();body.position=original.copy();body.position[axis]-=delta;negative=energy();body.position=original
                self.assertAlmostEqual(body.force[axis],-(positive-negative)/(2*delta),delta=1e-8)
                original=body.matrix.copy();direction=np.eye(3)[axis]
                body.matrix=rotation(direction,delta)@original;positive=energy()
                body.matrix=rotation(direction,-delta)@original;negative=energy();body.matrix=original
                self.assertAlmostEqual(body.torque[axis],-(positive-negative)/(2*delta),delta=1e-8)
        np.testing.assert_allclose(sum(body.force for body in bodies.values()),0.,atol=1e-14)
        np.testing.assert_allclose(sum(body.torque+np.cross(body.position,body.force) for body in bodies.values()),0.,atol=1e-14)

    def test_rank_one_split_preserves_complete_tensor_and_kinetic_energy(self):
        source=np.asarray([[5.,.3,.4],[.3,3.,.2],[.4,.2,2.]])*1e-4
        axis=np.asarray([1.,2.,3.]);axis/=np.linalg.norm(axis)
        shaft,remainder=inertia_partition(source,axis)
        self.assertGreater(min(np.linalg.eigvalsh(remainder)),0)
        np.testing.assert_allclose(source,remainder+shaft*np.outer(axis,axis),rtol=1e-14,atol=0)
        for velocity in ([1.,0.,0.],[.3,-.7,2.]):
            w=np.asarray(velocity)
            self.assertAlmostEqual(.5*w@source@w,.5*w@remainder@w+.5*shaft*(axis@w)**2,delta=1e-16)

    def test_torsional_coordinate_unwrap_is_accepted_step_bookkeeping(self):
        c=SimpleNamespace(ChVector3d=Vector);system=System()
        bodies={"a":Body([0,0,0],np.eye(3)),"b":Body([0,0,0],np.eye(3))}
        bodies["a"].omega=np.asarray([0.,0.,40.])
        spec=base("torsional_spring")
        declaration={"schema_version":"slim_equivalent_elasticity_v1","units":"m","elastic_reference_body_ids":[],"flexible_contact_regions":[],"elements":[spec]}
        handle=build_equivalent_elasticity(c,system,bodies,{},declaration)
        for step in range(1,81):
            handle.before_step(.01)
            bodies["a"].matrix=rotation([0,0,1],.4*step);system.time+=.01
            handle.after_step()
        row=handle.sample()["elements"]["test"]
        self.assertAlmostEqual(32.,row["coordinate"],places=12)
        self.assertAlmostEqual(.5*.3*32**2,row["potential_energy_j"],places=10)
        self.assertAlmostEqual(-.3*31.6,row["applied_step_generalized_force"],places=10)


if __name__ == "__main__":
    unittest.main()
