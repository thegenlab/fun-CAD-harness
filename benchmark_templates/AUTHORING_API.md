# Task-template authoring and instance-binding API

This page is read by the agents of a FunCAD-Harness run. It describes the
public template library in `benchmark_templates/` (the package name is kept for
compatibility with stored runs; it contains no benchmark tasks).

## Roles

A run has three distinct jobs. They must not be merged into one agent.

- **Constructor (author agent).** Builds and revises the mechanism from the task
  prompt. In free creation it also decomposes the prompt into task points and
  drafts `task_template.py` with the API below. It owns design trade-offs,
  allowed flexible/equivalent modelling and task difficulty; it requests
  difficulty changes by writing `template_adjustment.json`
  (`{"parameters": {...}, "reason": "..."}`).
- **Harness reviewer (independent agent).** Reviews the drafted template for
  prompt coverage, operations and checks, then writes the instance binding
  (`binding.py`) and `parameter_sources.json` for the current design. It
  implements the constructor's parameter decisions; disagreement is recorded as
  advice, never as a veto.
- **Harness (program).** Prepares the source model, checks that parts,
  operations and observation signals are connected, runs the physics and judges
  the fixed expressions. Physical results always come from the program.

Each revision repeats: simulate → feedback to the constructor → revised design
→ reviewer updates the binding → simulate again. Login, quota or connection
failures of an agent are kept as provider errors; they are never reported to the
constructor as mechanical defects.

## Template file

```python
from benchmark_templates.runtime import Parameter, Action, Check, Trial, Template, run_cli

PARAMETERS = (
    Parameter('stroke_m', 'target stroke', 'm', minimum=0.0),
    Parameter('force_n', 'maximum normal input force', 'N', minimum=0.0),
    Parameter('phase_s', 'duration of one operation phase', 's', minimum=0.0),
)

def build(p):
    t = p['phase_s']
    return Template(
        family_id='new_machine',
        source_task_ids=('T01',),
        required_signals=('output.position_m',),
        trials=(Trial(
            id='normal_operation', duration_s=t,
            actions=(Action('handle', 0.0, t, 'position', p['stroke_m'], p['force_n'], gain=100.0, damping=1.0),),
            checks=(Check('travel', ('T01',),
                          f"span(signal('output.position_m')) >= {p['stroke_m']}",
                          'the output reaches the target stroke after one operation'),),
            description='one complete operation; do not split it into unnecessary simulations',
        ),),
        description='how the mechanism is operated and judged',
    )

if __name__ == '__main__':
    run_cli(build, PARAMETERS)
```

The numbers above only illustrate the signatures. Operation constants such as
`gain` and `damping` should also be declared as `Parameter`s and filled from the
source design or the task.

### Fields

- `Parameter(name, description, unit, minimum=None, maximum=None, default=None,
  choices=None, owner='protocol')` — numeric, or an explicit `choices`
  enumeration. Parameters without a default must be filled by the instance
  protocol. The reviewer fills initial values and later applies the
  constructor's difficulty decisions, recording old value, new value and reason.
- `Action(port, start_s, end_s, mode, value, max_effort, gain=0.0, damping=0.0,
  min_effort=None, stop_at=None, stop_direction=1, bias_effort=0)` — modes
  `effort`, `speed`, `position` and `resistance`. For speed/position control the
  feed-forward bias and the feedback share one total effort limit and are both
  removed at the end of the action. `resistance` is a bounded drag opposite to
  the actual velocity (zero at zero speed); it does not model static locking.
- `Check(id, task_ids, expression, description, evidence='physics')` — a fixed
  expression over real signals. A requirement that genuinely needs a physical
  prototype or listening test uses `evidence='manual'` with an empty expression
  and is reported as pending manual evidence.
- `Trial(id, duration_s, actions, checks, description='', gravity=None,
  required_signals=(), sample_times=())` — every trial starts from a fresh
  instance of the same model. `gravity=None` keeps the source gravity; a
  non-negative scalar rescales it along the source direction; a 3-vector sets
  the world direction explicitly. Do not assume Z-down for a Y-up source.
- `Template(family_id, source_task_ids, required_signals, trials, description='',
  optional_signals=(), variant='default')` — mutually exclusive variants are
  selected by an enumerated protocol parameter; the template decides which task
  ids apply, never the binding.

Expressions use the Harness measurement syntax: `signal`, `min`, `max`, `mean`,
`span`, `rms`, `first`, `last`, `abs`, `integral`, `window`, `after`, `before`,
`cycles`, numbers and arithmetic comparisons. Arbitrary Python callbacks are not
accepted as judges; complex metrics use a real native measurement channel of
the binding. Several checks in one trial are evaluated from the same run.

## Instance binding

`binding.py` is written by the Harness reviewer, never by the constructor. It
exports `PROTOCOL` (the selected parameter values) and
`build_model(source_dir, step_s, protocol)`, or
`build_model_for_trial(source_dir, step_s, protocol, trial_id)` when trials need
genuinely different preparation (e.g. unloaded vs loaded). Use
`benchmark_templates.cad_native.build_native` on the supplied `prepared.json`.
The model exposes semantic signals and ports; the whole source mechanism stays
loaded and the template does not enumerate internal parts or require a
particular transmission layout. Optional parts the design does not have do not
affect required tasks; required functions cannot be dropped.

After each mechanical revision the reviewer re-checks the binding and may
update part references and design-dependent parameters chosen by the stated
rules. The original prompt, the required task types, the meaning of each check
and conditions the user fixed stay unchanged. Binding, execution or judging
errors are reported separately from mechanical failures; choosing a difficulty
never rewrites a physical result.

## Native model and update order

- `Signal` supports native axis coordinates/velocities, relative body
  positions/velocities and continuous angles/angular velocities about fixed
  axes. `source_scalar` only reads real state; it never returns an input target
  or an expected pass label.
- `Actuator` supports `shaft`, `body_force` and `body_torque`. The application
  point and velocity are the actual point of application. External hand or foot
  forces use `react_on_reference=False`.
- Each step: remove the previous manual effort → source internal forces →
  current manual operation → `commit_inputs` refreshes the solver force cache →
  one native step → source post-processing → sampling.
- Keep `commit_inputs` when loading with `cad_native`; without it a manual force
  can be applied one step late or not at all. Source callbacks and manual inputs
  use separate accumulators.

`startup_seconds` only runs a prefix and never certifies a full function. An
error in the program or an expression during a full run is reported as
inconclusive; only a value below the fixed criterion is a functional failure.
The CLI returns 2 for execution errors and 0 for completed tests, including
functional failures — read the saved state and evidence, not only the exit code.

`NativeModel.task_evidence_gaps` maps task ids to real missing physics evidence.
The runner keeps the numerical verdict in `numerical_passed`, but the task's
`passed` is `None` with status `unavailable_physical_evidence`.

Protocol times are rounded to 1e-12 s to avoid missing the last window through
accumulated floating-point error; physical thresholds are not rounded.

## Shared helpers

- `source_loading.build_authored_model(path, entrypoint, *args)` keeps the
  source factory and internal-force callbacks and isolates same-named Python
  modules of different designs. `install_authored_laws(path, entrypoint, model,
  ...)` installs internal laws in place.
- `actuators.PointForceTorquePort` applies a hand force at a real handle point
  with the current signed lever arm instead of torquing an output directly.
- `observations.observe_cycle_returns` checks return-to-start at real input
  cycle crossings, interpolating between samples.
- `observations.ContactMotionObserver` counts plucks only when a real contact
  and a measured displacement above `pluck_motion_min_m` coincide;
  `observations.CyclicPluckObserver` checks sequences from the source pin table.
- `runtime.build_trial_model(binding, source_dir, step_s, protocol, trial)`
  selects the per-trial instance; consecutive actions of one trial keep the same
  instance and history.
- `fixtures.prepared_with_source_fixture_joint` restores only an environment
  joint the source explicitly declares (e.g. an anti-sway test rig).
- `cad_native.prepare_declared_model_diagnostic` diagnoses a source that declares
  ideal constraints whose CAD support check failed; it records the support
  failure and the task evidence gap and never replaces the formal `prepare`.
- `motion.csv` keeps the requested operation in `input.<port>` and the actually
  applied generalized input in `input_actual.<port>`.
- `NativeModel.observation_events` stores contact events of a step and
  `native.run` writes them to `report.json`.

Some PyChrono `GetPos()`/`GetRot()` wrappers return live references. Copy the
xyz/wxyz values immediately when storing an initial state for displacement or
tilt checks.
