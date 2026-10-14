"""Implicit bounded external torque resistance for stationary-reference body ports.

The constitutive torque uses the step-average projected angular velocity. It
retains the nominal sliding torque but has no static lock at zero velocity.
Other port kinds keep their existing explicit resistance law and are identified
as such in the native protocol; they are not covered by this repair.
"""
import math


def torque_law(old_speed, trial_speed, inertia, step_s, bound):
    average = .5 * (old_speed + trial_speed)
    requested = -(2 * inertia / step_s) * average
    return min(bound, max(-bound, requested))


def _xyz(vector):
    return (float(vector.x), float(vector.y), float(vector.z))


def supported_port(actuator, operation):
    from .native import Actuator
    if type(actuator) is not Actuator or actuator.kind != "body_torque":
        return "Requires an unmodified native Actuator with kind=body_torque"
    if actuator.react_on_reference:
        return "Internal reaction-pair ports are outside the external single-body torque scope"
    if actuator.reference is not None and not actuator.reference.IsFixed():
        return "A moving reference requires a coupled-body load and is outside this scope"
    if operation.min_effort not in (None, -operation.max_effort):
        return "Asymmetric effort limits retain the existing explicit operation law"
    if actuator.obj.IsFixed():
        return "The resisted body must be dynamic"
    return None


def _load_type(ch):
    class MidpointResistance(ch.ChLoadCustom):
        def __init__(self, actuator):
            super().__init__(actuator.obj)
            self.body = actuator.obj
            axis = ch.ChVector3d(*actuator.axis)
            length = axis.Length()
            if not math.isfinite(length) or length <= 0:
                raise ValueError("Resistance axis must be finite and nonzero")
            axis = axis / length
            if actuator.reference is not None:
                axis = actuator.reference.GetRot().Rotate(axis)
            self.axis_world = _xyz(axis)
            self.old_speed = 0.0
            self.inertia = 1.0
            self.step_s = 1.0
            self.bound = 0.0
            self.active = False

        def IsStiff(self):
            return True

        def state(self, state_x=None, state_w=None):
            rotation = (ch.ChQuaterniond(*(float(state_x[i]) for i in range(3, 7)))
                        if state_x is not None else self.body.GetRot())
            local_axis = _xyz(rotation.RotateBack(ch.ChVector3d(*self.axis_world)))
            angular = tuple(float(state_w[i]) for i in range(3, 6)) if state_w is not None else _xyz(self.body.GetAngVelLocal())
            return local_axis, sum(a * w for a, w in zip(local_axis, angular))

        def prepare(self, operation, step_s):
            axis, self.old_speed = self.state()
            matrix = self.body.GetInertia()
            self.inertia = sum(axis[i] * float(matrix[i, j]) * axis[j] for i in range(3) for j in range(3))
            if not math.isfinite(self.inertia) or self.inertia <= 0:
                raise ValueError("Resistance needs positive finite inertia about its source axis")
            self.step_s = step_s
            self.bound = min(abs(operation.value), operation.max_effort)
            self.active = True

        def ComputeQ(self, state_x, state_w):
            # Only native state reads and this registered law occur inside Newton.
            # No model hooks, force commits, observations, or state setters run here.
            axis, speed = self.state(state_x, state_w)
            effort = torque_law(self.old_speed, speed, self.inertia, self.step_s, self.bound)
            self.load_Q.SetZero()
            for index, value in enumerate(axis):
                self.load_Q[index + 3] = effort * value

        def accepted_evaluation(self):
            self.ComputeQ(None, None)
            axis, speed = self.state()
            effort = sum(float(self.load_Q[i + 3]) * axis[i] for i in range(3))
            return effort, effort * .5 * (self.old_speed + speed)

    return MidpointResistance


class ResistanceLoads:
    def __init__(self, model, operations):
        candidates, legacy = {}, {}
        for operation in operations:
            if operation.mode != "resistance":
                continue
            actuator = model.actuators[operation.actuator]
            reason = supported_port(actuator, operation)
            if reason:
                legacy[operation.actuator] = reason
            else:
                candidates[operation.actuator] = actuator
        # A port with any unsupported operation keeps one consistent legacy path.
        candidates = {name: actuator for name, actuator in candidates.items() if name not in legacy}
        self.loads, self.power = {}, {}
        self.contract = {"implicit_ports": list(candidates), "legacy_explicit_ports": legacy}
        if not candidates:
            return
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import load_pychrono_core
        ch = load_pychrono_core()
        solver = ch.ChSolverADMM()
        solver.SetMaxIterations(300)
        solver.SetTolerancePrimal(1e-12)
        solver.SetToleranceDual(1e-12)
        model.system.SetSolver(solver)
        stepper = ch.ChTimestepperEulerImplicit(model.system)
        stepper.SetMaxIters(30)
        stepper.SetAbsTolerances(1e-12, 1e-12)
        model.system.SetTimestepper(stepper)
        container = ch.ChLoadContainer()
        model.system.Add(container)
        load_type = _load_type(ch)
        for name, actuator in candidates.items():
            load = load_type(actuator)
            container.Add(load)
            self.loads[name] = load
            self.power[name] = 0.0
        self.keepalive = (container, solver, stepper)
        self.solver, self.stepper = solver, stepper
        model._passive_resistance_keepalive = self
        self.contract.update(
            law="torque = -sat((2 * axis_inertia / step_s) * (old_speed + trial_speed) / 2, nominal_bound)",
            regularization="Finite step-average viscous center; no static lock; unchanged nominal sliding torque",
            scope="External torque on one dynamic body; world axis or fixed reference; no coupled reference reaction",
            inertia="Source body's full local inertia tensor projected onto the world brake axis at step start",
            solver={"type": "ADMM", "max_iterations": 300, "primal_tolerance": 1e-12, "dual_tolerance": 1e-12},
            integration={"type": "EulerImplicit", "max_iterations": 30, "state_absolute_tolerance": 1e-12,
                         "multiplier_absolute_tolerance": 1e-12, "scope": "Entire native run containing these loads"},
            actual_effort="Accepted-state implicit-load constitutive evaluation after a successful solver step; not a GetQ measurement",
            qualification="Requires convergence and independent momentum/energy residual validation; no post-step state or power correction",
            average_power="Evaluated torque times the step-average projected angular velocity",
        )

    def clear(self):
        for name, load in self.loads.items():
            load.bound = 0.0
            load.active = False
            load.load_Q.SetZero()
            self.power[name] = 0.0

    def prepare(self, name, operation, step_s):
        self.loads[name].prepare(operation, step_s)

    def accepted(self, actuators):
        if self.loads:
            solves = self.stepper.GetNumStepSolveCalls()
            diagnostics = {"newton_iterations": self.stepper.GetNumStepIterations(),
                           "admm_solve_calls": solves,
                           "admm_iterations": self.solver.GetIterations(),
                           "primal_error": float(self.solver.GetErrorPrimal()) if solves else None,
                           "dual_error": float(self.solver.GetErrorDual()) if solves else None}
            self.contract["last_step_convergence"] = diagnostics
            if (diagnostics["newton_iterations"] >= self.stepper.GetMaxIters()
                    or (solves and (not math.isfinite(diagnostics["primal_error"]) or not math.isfinite(diagnostics["dual_error"])
                        or diagnostics["primal_error"] >= self.solver.GetTolerancePrimal()
                        or diagnostics["dual_error"] >= self.solver.GetToleranceDual()))):
                raise RuntimeError("Implicit resistance solve did not establish convergence: " + str(diagnostics))
        for name, load in self.loads.items():
            if not load.active:
                continue
            effort, power = load.accepted_evaluation()
            actuators[name]._last_input = effort
            self.power[name] = power
