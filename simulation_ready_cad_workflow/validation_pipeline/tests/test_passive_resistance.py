"""Pure-law and fake native-interface regressions; not physical validation."""
import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark_templates import native
from benchmark_templates.passive_resistance import ResistanceLoads, _load_type, torque_law


class Vector:
    def __init__(self, x=0., y=0., z=0.): self.x, self.y, self.z = x, y, z
    def Length(self): return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def __truediv__(self, value): return Vector(self.x/value, self.y/value, self.z/value)


class Quaternion:
    def __init__(self, w=1., x=0., y=0., z=0.): self.e0, self.e1, self.e2, self.e3 = w, x, y, z
    def Rotate(self, vector):
        w, x, y, z = self.e0, self.e1, self.e2, self.e3
        return Vector((1-2*y*y-2*z*z)*vector.x + (2*x*y-2*w*z)*vector.y + (2*x*z+2*w*y)*vector.z,
                      (2*x*y+2*w*z)*vector.x + (1-2*x*x-2*z*z)*vector.y + (2*y*z-2*w*x)*vector.z,
                      (2*x*z-2*w*y)*vector.x + (2*y*z+2*w*x)*vector.y + (1-2*x*x-2*y*y)*vector.z)
    def RotateBack(self, vector): return Quaternion(self.e0, -self.e1, -self.e2, -self.e3).Rotate(vector)


class Matrix:
    def __init__(self, values): self.values = values
    def __getitem__(self, index): return self.values[index[0]][index[1]]


class Body:
    def __init__(self, speed=.002, fixed=False):
        self.speed, self.fixed, self.reads = speed, fixed, 0
        self.inertia = Matrix([[.002, 0, 0], [0, .002, 0], [0, 0, .002]])
    def IsFixed(self): return self.fixed
    def GetRot(self): self.reads += 1; return Quaternion()
    def GetAngVelLocal(self): self.reads += 1; return Vector(0, 0, self.speed)
    def GetAngVelParent(self): return self.GetAngVelLocal()
    def GetInertia(self): return self.inertia


class LoadVector(list):
    def SetZero(self): self[:] = [0.] * 6


class CustomLoad:
    def __init__(self, body): self.load_Q = LoadVector([0.] * 6)
    def GetQ(self): raise AssertionError('Unsafe GetQ binding must never be called')


class Solver:
    def __init__(self): self.primal_error = self.dual_error = 0.
    def SetMaxIterations(self, value): self.maximum = value
    def SetTolerancePrimal(self, value): self.primal = value
    def SetToleranceDual(self, value): self.dual = value
    def GetMaxIterations(self): return self.maximum
    def GetIterations(self): return 2
    def GetErrorPrimal(self): return self.primal_error
    def GetErrorDual(self): return self.dual_error
    def GetTolerancePrimal(self): return self.primal
    def GetToleranceDual(self): return self.dual


class Stepper:
    def __init__(self, system): self.iterations = 2
    def SetMaxIters(self, value): self.maximum = value
    def SetAbsTolerances(self, *values): self.tolerances = values
    def GetMaxIters(self): return self.maximum
    def GetNumStepIterations(self): return self.iterations
    def GetNumStepSolveCalls(self): return 2


class Container:
    def __init__(self): self.loads = []
    def Add(self, load): self.loads.append(load)


FAKE_CHRONO = SimpleNamespace(ChLoadCustom=CustomLoad, ChVector3d=Vector, ChQuaterniond=Quaternion,
                             ChSolverADMM=Solver, ChTimestepperEulerImplicit=Stepper, ChLoadContainer=Container)


class System:
    def __init__(self, body, events, *, fail_convergence=False):
        self.body, self.events, self.time, self.fail_convergence = body, events, 0., fail_convergence
    def SetSolver(self, solver): self.solver = solver
    def SetTimestepper(self, stepper): self.stepper = stepper
    def Add(self, container): self.container = container
    def GetChTime(self): return self.time
    def DoStepDynamics(self, dt):
        self.events.append('step')
        for load in self.container.loads:
            load.ComputeQ([0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, self.body.speed])
            load.ComputeQ([0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        self.body.speed = 0.  # Fixture solution for this isolated unsaturated step, not a production setter.
        if self.fail_convergence: self.solver.primal_error = 1.
        self.time += dt
        return True


class PassiveResistanceTests(unittest.TestCase):
    def operation(self): return native.Operation('brake', 0, .001, 'resistance', .9, .9)
    def actuator(self, body): return native.Actuator(body, 'body_torque', react_on_reference=False)

    def test_nominal_torque_coasting_reversal_and_no_static_lock(self):
        self.assertEqual(torque_law(10, 9, .002, .001, .9), -.9)
        self.assertEqual(torque_law(0, 0, .002, .001, .9), 0)
        self.assertAlmostEqual(torque_law(.002, 0, .002, .001, .9), -.004)
        torque = torque_law(.02, -.350812, .002, .001, .9)
        self.assertGreater(torque, 0)
        self.assertLess(torque * (.02-.350812)/2, 0)
        new_speed = .2 * .001 / (2 * .002)
        self.assertGreater(new_speed, 0)
        self.assertAlmostEqual(.002 * new_speed, .001 * (.2 + torque_law(0, new_speed, .002, .001, .9)))

    def test_computeq_uses_trial_velocity_and_orientation_without_body_callbacks(self):
        body = Body()
        load = _load_type(FAKE_CHRONO)(self.actuator(body))
        load.prepare(self.operation(), .001)
        body.reads = 0
        load.ComputeQ([0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, .004])
        self.assertAlmostEqual(load.load_Q[5], -.012)
        half = math.sqrt(.5)
        load.ComputeQ([0, 0, 0, half, 0, half, 0], [0, 0, 0, .004, 0, 0])
        self.assertAlmostEqual(load.load_Q[3], -.004)
        self.assertAlmostEqual(load.load_Q[5], 0)
        self.assertEqual(body.reads, 0)
        self.assertTrue(load.IsStiff())

    def test_axis_inertia_uses_off_diagonal_source_tensor(self):
        body = Body()
        body.inertia = Matrix([[2, .5, 0], [.5, 4, 0], [0, 0, 1]])
        actuator = self.actuator(body); actuator.axis = (1, 1, 0)
        load = _load_type(FAKE_CHRONO)(actuator)
        load.prepare(self.operation(), .001)
        self.assertAlmostEqual(load.inertia, 3.5)

    def test_other_ports_are_explicitly_legacy_without_solver_changes(self):
        actuator = native.Actuator(Body(), 'body_force', react_on_reference=False)
        model = SimpleNamespace(actuators={'brake': actuator}, system=object())
        loads = ResistanceLoads(model, [self.operation()])
        self.assertFalse(loads.loads)
        self.assertIn('brake', loads.contract['legacy_explicit_ports'])

    def test_inactive_resistance_does_not_replace_other_input_on_same_port(self):
        body = Body()
        actuator = self.actuator(body)
        model = native.NativeModel(System(body, []), {}, {'brake': actuator})
        with patch('simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core.load_pychrono_core', return_value=FAKE_CHRONO):
            loads = ResistanceLoads(model, [self.operation()])
        loads.clear()
        actuator._last_input = .4
        loads.accepted(model.actuators)
        self.assertEqual(actuator._last_input, .4)
        self.assertEqual(loads.power['brake'], 0)

    def test_runner_syncs_accepted_effort_before_after_step_and_records_mean_power(self):
        body, events = Body(), []
        actuator = self.actuator(body)
        def after(now, dt):
            self.assertAlmostEqual(actuator._last_input, -.004)
            events.append('after')
        model = native.NativeModel(System(body, events), {'speed': native.Signal(body, 'body_angular_speed')},
            {'brake': actuator}, before_step=lambda *args: events.append('before'),
            commit_inputs=lambda *args: events.append('commit'), after_step=after)
        with tempfile.TemporaryDirectory() as temporary, patch(
                'simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core.load_pychrono_core', return_value=FAKE_CHRONO):
            output = Path(temporary) / 'run'
            result = native.run(model, [self.operation()], duration_s=.001, step_s=.001, output=output)
            with (output / 'motion.csv').open() as stream: rows = list(csv.DictReader(stream))
            protocol = json.loads((output / 'protocol.json').read_text())
        self.assertEqual(events, ['before', 'commit', 'step', 'after'])
        self.assertEqual(result['status'], 'completed')
        self.assertAlmostEqual(float(rows[-1]['input_actual.brake']), -.004)
        self.assertAlmostEqual(float(rows[-1]['input_average_power.brake']), -.000004)
        self.assertIn('not a GetQ measurement', protocol['passive_resistance']['actual_effort'])
        self.assertEqual(actuator._last_input, 0)

    def test_nonconverged_native_step_fails_before_source_after_step(self):
        body, events = Body(), []
        actuator = self.actuator(body)
        model = native.NativeModel(System(body, events, fail_convergence=True), {}, {'brake': actuator},
                                   commit_inputs=lambda *args: None, after_step=lambda *args: events.append('after'))
        with tempfile.TemporaryDirectory() as temporary, patch(
                'simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core.load_pychrono_core', return_value=FAKE_CHRONO):
            output = Path(temporary) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'did not establish convergence'):
                native.run(model, [self.operation()], duration_s=.001, step_s=.001, output=output)
            report = json.loads((output / 'report.json').read_text())
        self.assertEqual(report['status'], 'execution_error')
        self.assertNotIn('after', events)


if __name__ == '__main__':
    unittest.main()
