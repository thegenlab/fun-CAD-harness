# Slim Harness

The existing direct-scene entrypoint accepts author-delivered `scene.json`,
geometry, and a proposed test set. Any modeling tool is allowed. There is no mandatory mechanism family, joint catalog, task template,
planning identity, or role protocol to satisfy. The Harness prepares original
geometry, runs Chrono and program checks, measures the actual states, and returns
evidence. Model review is limited to task/test changes as described below.
The existing validator is a separate path.

The free-creation mode (see the [template authoring API](../../../benchmark_templates/AUTHORING_API.md))
requires a distinct initial step: for a new mechanical prompt, the construction
agent decomposes the required functions and authors a new task template; the
Harness review agent reviews that template and fills its design-specific
parameters and operations. Simulation feedback then drives mechanical/template
revision and reviewer-maintained binding updates. The constructor may author
the template but must not review it or fill its own instance binding.
Select `create --mode free` to invoke that initial template-review/fill sequence.
The older direct-scene entrypoint's lack of initial review does not apply to
this explicitly selected template mode.

The constructor owns tradeoffs between mechanical design, permitted
flexible/equivalent models and task difficulty, and may request simpler
concrete test parameters through `template_adjustment.json`; the Harness review
agent implements those binding changes and records their basis, and its advice
is not a veto. The original prompt, check meaning and explicit user
restrictions are preserved, the actual tested target is reported, and program
physics remains authoritative.

## Run

From this repository, use the CAD environment's Python (`python` below). Workflow files and criteria are
created by the program; the author does not prepare them.
The slim author retains configured app/plugin tools as well as shell access;
the Harness does not limit it to the CAD generator used by these pilots.

```bash
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness create --mode free --project my_machine --task task.md
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness submit path/to/scene.json --mode free --project submitted_machine --task task.md
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness resume my_machine
```

New ordinary CLI `create` / `submit` tasks default to the existing **high** tier: up to 5 test tasks and 20 physical parts, with the tier's recommended active work time and per-simulation cap (`budget.recommended_limits`: 3 hours and 300 seconds, unless the distribution's `execution_profile.py` recommends other values; `--budget-hours` and `--simulation-timeout` override them). Explicit `--tier` overrides it; `resume` retains the saved tier. Default high `create` now uses supervised construction with one activity-union clock for authors, CAD and audit; simulation wall time is not charged. Old projects and explicit light/medium routes retain their saved behavior. `submit` remains a one-revision evaluation unless `--construction` is requested. Construction authors share the remaining project time, minus dynamically reserved first-test time, without an additional 1800-second turn ceiling.

New Codex projects default to author `gpt-6-astra` with `high` reasoning. Cursor
keeps its existing default author model and reasoning metadata; no unsupported
Cursor reasoning flag is added. New projects using either author select an
independent Codex reviewer, `gpt-6-astra` with `medium` reasoning, unless reviewer
options are explicit. All selected author and reviewer settings are persisted.
Resume preserves each project's saved choice, including `medium`; a historical
project without a saved reasoning field retains its previous `medium` fallback.
Missing reviewer settings in historical projects are not silently rewritten.

`create` builds and revises autonomously. `submit` evaluates one existing
delivery and returns feedback; it copies only referenced assets into an isolated
revision. `resume` continues a project. A natural-language `--brief` can replace
the task file. Existing project names are never overwritten. Tool invocations
have finite process guards; there is no implicit one-hour stopping budget.
Provider and Harness faults are reported separately from design feedback.
With a budget tier, a failed attempt or a single simulation timeout returns
the current remaining active time and concrete evidence to the author. The
author records its own choice: revise the mechanism, change an allowed
execution/model choice, propose a task adjustment, or finish with partial
results. No rule assigns the failure type or selects an action from a budget
percentage. Each decision retains its reason, tradeoff and pre-turn budget.

For an open task that may be relaxed, select a budget tier and pass `--adjustable` on `create` or
`submit`. The author may then propose `task_adjustment.json` with `task_text`,
`reason` and optional source-quoted `criteria`, together with the complete proposed
tests in `scene.json`. One task/test-change review advises on this combined change; the author
decides how to proceed. The program retains the original task/criteria/lock bytes
and the advice under `task_adjustments/`, and binds any applied criteria changes.
After the author accepts the change, physics and sampled source audit run
under the program gates. Results explicitly identify
the adjusted target and never claim that the original requirement passed.
`--adjustable` cannot be combined with a frozen benchmark `--experiment`.
Existing projects remain non-adjustable unless explicitly authorized as such.
Recovery reuses expensive results only when their inputs and the relevant
calculation implementation still match. A code repair that changes geometry,
physics, measurements, or geometry audit invalidates the affected recovery cache.

### First-pass test coverage

Each declared independent test receives an execution opportunity before author repair. A functional failure, per-test timeout or runtime error is recorded and does not end the rest of the round. All physical launches precede acceptance-only motion auditing and replay work. Shared scene/geometry/binding failures and actual total-budget exhaustion remain stopping conditions.

Previously untested definitions run first. An unchanged failed trace can be reused as failed evidence without blocking later tests. Resource retries cannot delay a later test's first opportunity. Draft `simulate` returns `FIRST_PASS_PENDING` if the selected definition was already attempted while other declared definitions remain untested.

Author time excludes a reserved first-pass simulation allowance (up to one per-simulation cap per pending test, e.g. 5 × 300 seconds for high). When this allowance is reached, the program evaluates the latest runnable scene instead of spending the remaining time on more author retries. If no valid scene exists it records the shared blocker. An interrupted partial delivery does not directly grant whole-machine PASS; a subsequent author-ready delivery is still required.

`first-pass-coverage.json` schedules work only; it is not a certificate or a cache of acceptance. Per-attempt `test-coverage.json` distinguishes visited, started/reused and not-run tests. `pending_tests` can include a timed-out test awaiting a complete trace; `unrun_tests` identifies tests without a confirmed execution. Total time limits, frozen criteria, numerical settings and acceptance gates remain unchanged.

### Existing frozen concrete-test interface

Normal `create` establishes the initial task and first `scene.tests` as the
baseline without model review. To freeze a concrete declarative test definition,
pass `--experiment experiment.json` to either `create` or `submit`. This uses the
same program checks, with researcher-supplied criteria and tests:

```json
{
  "schema_version": "slim_frozen_experiment_v1",
  "task_text": "The slider must travel at least 10 mm.",
  "units": "mm",
  "gravity_m_s2": [0, -9.81, 0],
  "reviewer_summary": "Researcher-frozen stroke test",
  "criteria": [{"id": "stroke", "kind": "function",
    "statement": "Travel at least 10 mm", "source_quote": "at least 10 mm",
    "verification": "Measure slider travel under the declared input"}],
  "tests": [{"id": "stroke", "purpose": "Measure travel", "duration_s": 1,
    "inputs": [{"joint": "drive", "mode": "effort", "value": 1, "max_effort": 1}],
    "assertions": [{"id": "travel", "criterion_ids": ["stroke"], "purpose": "Travel",
      "expression": "span(signal('slider.position.x_m')) >= 0.01"}]}]
}
```

`task_text` must match the task file exactly. The program copies and binds this
file to the project; subsequent author revisions cannot change its tests, units,
gravity or criteria. The author keeps the test's required body/joint IDs and may
omit `scene.tests`, which the program supplies. Changed test definitions require
a new experiment. Frozen tests are not a PASS: physical execution, source checks
and the other program gates remain required.
This is the Slim declarative test format, not the earlier native runner's executable `experiment.py`.
Its fixed body/joint references are specific to this interface. The
free-creation template loop described above instead has the reviewer fill the
design-specific references.

## Delivery

`scene.json` uses `schema_version: "slim_scene_v1"` and `units: "mm"` or `"m"`.
Geometry is already placed in its assembled world coordinates. STEP units are
read from STEP; `units` applies to JSON coordinates and unitless mesh files.

* `bodies`: `{id, source, solid_ids?, fixed?, density_kg_m3?, color?}`.
  Source is a relative STEP/STP, STL, OBJ, or GLB file. `solid_ids` are
  `solid_1`, `solid_2`, ... in source import order; omit to use the whole file.
  Every source solid belongs to exactly one body. One STEP file per body is
  also accepted. Meshes must be closed solids. The program derives center of
  mass and inertia; density defaults to 7800 kg/m鲁. Fixed bodies represent
  an explicitly declared external mounting or workbench with a stated basis.
* `joints`: `{id, type, body_a, body_b, anchor, axis, region_a, region_b}`.
  Native helpers currently implement `revolute`, `prismatic`, `spherical`,
  and `fixed`. `anchor` is a world point; `axis` is a direction (unitless).
  A region is `{points: [[x,y,z], [x,y,z]], radius: number}`: a body-attached
  capsule, or a sphere when the two endpoints coincide. These are finite
  connection regions near real material, not penetration exemptions. Only
  declared pairs connect; proximity never creates a joint. When regions cease
  to overlap (including a 0.05 mm numerical gap), the native link and its motor
  are disabled before the next step. A disabled broken joint stays released.
  Use short regions for a bearing and finite rail-length regions for a slider.
  Optional nonnegative `damping` is passive viscous damping in N路m路s/rad
  (revolute) or N路s/m (prismatic), active only inside the connection region.
* `tests`: scenarios described below. `gravity_m_s2` defaults to `[0,-9.81,0]`.
  All dynamics use SI internally. Materials are provided by the Harness:
  friction 0.3, restitution 0, with the exact settings recorded in the trace.

Optional revolute `support` evidence contains `shaft` and `bore` objects with
`body`, world `center`, `axis`, `radius`, `length`, plus `max_clearance` on
`support`. Lengths follow scene units. The program checks these features against
the source surfaces/material and the declared joint axis. Radial support alone
does not establish axial retention; the author must explain its actual source
geometry. Program source checks remain required.

### Equivalent external spur gears

Optional `scene.spur_gears` declares parallel-axis external spur pairs with
finite native tooth stiffness and damping. Each pair names its two dynamic
bodies, their revolute bearings to fixed material, common world axis and shaft
centers, tooth counts (at least 17), module, face width, pressure angle, stiffness,
damping and a parameter basis. Quantities whose names end in `_m`, `_rad`,
`_n_m` or `_n_s_m` use SI units regardless of `scene.units`.

The compiler checks source radial tooth profiles and an axial material section
through every tooth against the declared centered face width. This limited
straight-tooth check rejects thin offset or separated tooth bands; it is not
a complete involute conformity certificate. Runtime replaces contacts only when
both contact points lie in the active pair's finite tooth annuli. Bearing
release, excessive center spacing/axis tilt, or loss of face overlap disables
the pair and restores native contact. Release is permanent for that test.

The native compliant law and sampled Coulomb loss report finite forces and
dissipation. They do not resolve individual tooth contact, backlash, impact,
stress or deformation. The author is responsible for whether the model and its
parameters answer the task. Original-geometry motion audit remains required;
geometry or model changes alone do not trigger model review. This feature leaves
existing mesh preparation, contact margins, solver settings and print-delivery
verification unchanged.

See `test_slim_spur_gears.nominal()` for the declaration fields. Each completed
trace records its declaration, exclusions, forces, releases and model scope;
published results retain that scope rather than claiming detailed tooth contact.

### Numerical recovery

Original STEP/mesh collision, mass, inertia, display and motion audit are retained.
STEP triangulation uses 0.05 mm absolute deflection and a 0.12 radian angle,
with OCCT interior deflection control enabled for all surface types. This
includes analytical cylinders: edge deflection alone can leave long interior
diagonals outside the target on a trimmed face. Mass and inertia still come
from the original BRep. The user-selected 0.05 mm target is shared by mesh
preparation and its geometry regressions. It is a meshing target, not a universal
accuracy certificate: the original BRep and every source component remain subject to
the uniform 10 Hz saved-pose audit with exact trace endpoints. The full native
trace remains available; no claim is made about unselected poses or continuous
clearance. Unchanged relative body pairs reuse their successful check within
the same source-bound audit, while new relative motion or changed local
exclusion activity invalidates that reuse.
Source-support checks use `rtree` in the CAD environment; the Chrono worker does
not import CAD geometry libraries. The proposed native collision switch
is not enabled: local thin-support contact checks failed. See the integration
record for the saved candidate and diagnostics.

Rigid source triangles have zero sphere-swept thickness. After Bullet initializes
their collision models, the runtime restores and reads back a 100 micrometre
safe margin per body; `source_mesh_collision_margins` records the effective values.
Connected-triangle initialization otherwise overwrites that margin with the swept
radius and rejects initially touching or slightly penetrating surface contacts.
The triangle search envelope is also 100 micrometres per body. Both search
windows extend beyond the unchanged 50 micrometre penetration rejection so
an overshot contact can be reported and trigger a smaller-step retry instead
of disappearing from detection. The original surfaces, initial poses and
mass properties are unchanged; positive-gap candidates still use the native
NSC gap/step constraint rather than moving the source surfaces.

Source facets longer than 80 mm receive conformal midpoint subdivision; their
surface, winding and cavities remain unchanged. Long rail triangles otherwise
lost tangent capsule/sphere contacts in the native detector. For original STEP
solids, preparation also recognizes coaxial cylindrical faces and checks the
entire derived axial interval with a non-destructive OCCT material subtraction.
Only a completed cut with zero missing volume permits an inscribed capsule of
the same radius and outer length. It supplements the complete source mesh;
it does not replace the body, fill a bore, change mass properties, or select
support points using a test or another body. These Harness-derived primitives
and their original-solid proofs are recorded in `source_analytic_material`.

For an ambiguous near-coincident contact, the runtime can reconstruct its points
and normal from the original surfaces: one matched source vertex, an in-face
projection, a unique pair of opposite face normals, and source planes that
support both local neighborhoods are required. Source gaps exceed neither the
10 nm native coordinate band nor the unaltered penetration limit. Ambiguous
cases remain native; in particular, a tangent horizontal face does not hide a
real side-wall penetration at a reflex corner. Clear outward constraints on
convex source vertices are filtered separately, with deep material witnesses
retained for rejection. Both operations precede physical acceptance and retain
the independent original-STEP motion audit. Their counts and bounded geometric
witnesses are stored in the trace.

PSOR uses its fixed sweep budget rather than its within-sweep stopping estimate.
The latter can report zero before later coupled contacts restore a violation:
an initially touching calibration box then receives less than its weight in
support force. The normal budget remains 160 sweeps (240 for native friction
clutches); this change does not raise the allowed penetration or alter loads.

Rigid steps use at most 1 ms or the smaller requested step. The workflow CLI
also accepts `--physics-max-step-seconds`, positive and finite and no greater
than 0.001 (default). This Harness-owned limit is recorded in the request's
top-level `numerical_settings`; it does not edit the author's `test.step_s`.
The effective initial step is the minimum of 1 ms, the author's upper bound
and the workflow limit. Recorded contact
penetration, nonconvergence or nonfinite solved states may retry with half the
step, at most twice within the native process budget (600 seconds by default).
For longer tests, the workflow CLI accepts `--physics-timeout-seconds`; the
positive finite limit covers the whole test process and all its retries. It is
recorded in the request's `execution_limits`. The default memory policy is
`auto`: use 80% of currently available host/container RAM, leave 20% initial
headroom, and tighten the owned process-tree allowance if competing work
consumes that headroom. Linux cgroup v1/v2 and visible ancestor limits bound
the host reading. Explicit project memory overrides remain effective. Actual
limits are recorded in `memory-budget.json` or construction/probe/audit cost
reports, including timeout and RSS failures. This is a polling guard, not an
operating-system reservation of RAM. This resource setting does not change the scene, test or acceptance
criteria, and an exhausted budget remains a Harness error. Each trace is
retained, with the actual step and total evaluation time. Result metadata records
`requested_max_step_s` and `effective_initial_step_s`. A 0.5 ms workflow limit
starts at 0.5 ms and permits the same two halvings to 0.25 and 0.125 ms.
Every physical cache shortcut, including completed results and interrupted
failure recovery, must match this effective initial-step policy; matching only
the last retry's step is insufficient. Loads, duration and
assertions never change. Ordinary task failures do not trigger numeric retries;
exhausted numerical failures remain unresolved Harness errors, not instructions
to redesign the mechanical source. Step retries do not claim numerical convergence.

Bodies connected through currently active fixed joints form one rigid component.
Native collision families omit contacts within that component while retaining
every cross-component contact. Released fixed links update the groups before
the next solver step; original meshes, mass, inertia and joint reactions remain.
This does not exempt original geometry from the initial or motion audit.

These helpers describe physical behavior, not an admission taxonomy. A needed
behavior without an implementation is reported as a Harness capability gap;
the author does not have to redesign a machine to imitate another family.

## Equivalent elastic behavior

Source-bound finite external spur-gear transmission is available through
`scene.spur_gears`; see [Spur gears](SPUR_GEARS.md) for supported geometry,
finite engagement, friction approximation and backlash limitations.

For an explicitly selected equivalent elastic model, add the source-bound
`equivalent_elasticity` declaration described in [Equivalent elasticity](EQUIVALENT_ELASTICITY.md).
It supports two-ended springs, Coulomb friction and original-STEP detent profiles.
The result carries its narrower geometry/material scope through physics, audit
and publication. The complete required functional test set still runs.

## Proposed tests

Each test has `id`, `purpose`, `duration_s`, `step_s` (default 0.001), `inputs`,
`loads`, and `assertions`. Tests begin from the delivered assembled state.
Input and body-load activity uses the half-open interval `[start_s, end_s)`.
Effort and applied-force samples at a step endpoint describe the just-completed
step: an input ending at 28 s is absent from the step starting at 28 s, whose
effort is recorded at 28.001 s for a 1 ms step.

An input drives one declared joint with finite effort:
`{joint, mode: "speed"|"effort"|"position", value, max_effort, gain?, damping?,
start_s?, end_s?}`. Speed and position are rad/s and rad for revolute joints,
m/s and m for prismatic joints; effort is N路m or N. Speed/position controls are
bounded force controllers, not prescribed output trajectories. Only intended
inputs may be driven. Optional `wave` is `{amplitude, frequency_hz, offset?}`
and replaces `value` by an SI sinusoid. Loads are
`{body, force_n?, torque_nm?, point?, start_s?, end_s?}` in world coordinates;
omitting `point` applies force at the center of mass. Loads must match the
test purpose and must not secretly assist the measured output.

An assertion is `{id, criterion_ids, purpose, expression}`. Expressions use
real recorded signals with `signal("name")`, arithmetic, comparisons, and
`min`, `max`, `span`, `mean`, `rms`, `first`, `last`, `abs`, `integral`,
`after(values, time_s)`, `before(values, time_s)`, `window(values, start_s, end_s)`,
and `cycles(values, minimum_stroke, return_tolerance)`. Scalars and numeric arrays
are supported. Examples:

```
span(signal("piston.position.x_m")) >= 0.04
abs(last(signal("beam.angle.z_rad"))) <= 0.03
max(abs(after(signal("beam.angular_speed.z_rad_s"), 2.0))) <= 0.02
cycles(signal("piston.position.x_m"), 0.030, 0.001) >= 2
```

Signals include each body's `position.{x,y,z}_m`, `angle.{x,y,z}_rad`
(integrated world-axis angular travel, suitable for fixed-axis mechanisms),
`linear_speed.{x,y,z}_m_s`, `angular_speed.{x,y,z}_rad_s`, each joint's
`coordinate` and `active`, `input.<joint>.effort`, `input.<joint>.work_j`,
`contact.count`, `contact.max_penetration_m`, and `time_s`.
`body_id.orientation_error_rad` measures actual quaternion orientation change
from the initial pose and is suitable for general 3-D return tests. Feedback
controllers use derived inertia to stabilize their sampled force calculation;
the actual finite effort and discrete mechanical work are recorded.
For fixed-axis rotation prefer a joint's continuous `coordinate`. No task
template ID is required; these are optional measurement functions.
`cycles` counts non-overlapping consecutive extremum/opposite-extremum/return
triples, requires both legs to meet the stroke and the endpoint to return within
tolerance, and does not add partial reversals together. A clipped boundary
cycle is not extrapolated. `window` preserves time coordinates for integration.
The automatic geometry preparation reads only declared source files; admission
does not require running the legacy catalog-dependent inspection command.

## Review and evidence ownership

The initial task and first `scene.tests` establish the baseline without model
review. A later change to the task, `scene.tests` or default gravity triggers one advisory review
of that change's reasonableness. Geometry-only changes do not trigger it.
There are no initial, visual, static-preflight or final model reviews.

The review limit is 60/120/180 seconds for light/medium/high. At that limit the
program sends one fixed reminder, allows at most 60 additional seconds, then
hard-stops the scoped reviewer. Available output, including timeout or failure,
returns to the author, who decides whether to amend the proposal or continue.
Advice does not approve a design or override a program gate.
For Codex CLI, the reminder interrupts and resumes the same independent review
session; it does not start a new review. Identical pending changes reuse their
saved advice. The project budget can stop the review earlier.

The program owns criteria persistence, physical execution, signals, source
geometry audit and program verdicts. Frozen conditions and hard checks remain
enforced. Preview images cannot establish physics.
Collider meshes contain the original surface, with no decimation or convex-hull
replacement. Saved-motion geometry audit checks uniform 10 Hz states and exact endpoints; the report states
that resolution explicitly and does not claim a proof between solver steps.
Failed revisions and provider/infrastructure errors are retained separately.

### Executed print delivery evidence

`print_delivery` can register source/export evidence for exact frozen criteria
without assigning a motion test to an STL or booklet requirement. A supervisor
supplies the request outside the author design. Its `parts` map each physical
instance to its scene body (or a separate coupon), STEP, STL, quantity, and
4x4 source-to-print transform in millimetres; `plates` map those STLs to their
actual layout transforms. The request also binds all source files, a fresh
modeling build entry point, assembly STEP/GLB, quantity list, print dimensions,
and documentation including local HTML images.

```bash
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness.print_delivery --project <project> --design <revision-design> --request <supervisor-request.json> --out <new-evidence-directory> --cad-python "$SIMREADY_CAD_PYTHON" --register
```

The bounded worker rebuilds the delivered modeling source, opens original and
rebuilt STEP/STL geometry, and checks instance correspondence, connected parts,
print envelopes, layout spacing, and current GLB/layout exports. Tessellation
checks use vertex/edge/face samples within 0.05 mm and volume within 0.5 percent;
they are not continuous BRep or physical fidelity certificates. Every receipt
is revalidated against current task, source, implementation, runtime and outputs.

`static_only` coverage is for exclusively static delivery criteria and requires
the corresponding program evidence. `supplemental` covers
the static part of mixed criteria and retains required motion-test coverage.
The author is responsible for actual assembly, illustrations and document
semantics; static delivery adds no model review. Mechanical tests and original-
geometry audits remain required. The existing exact alarm F7 verifier and its
registration are retained unchanged.

## Joint regions and force overlays

Original-shape motion replay can show each joint's two body-attached capsule
regions, recorded world-space reaction forces/torques, contact wrenches and
applied loads. The viewer provides separate region, heatmap and force-path
switches, a joint selector, and separate N / N路m heat scales. These are finite
activation regions for ideal rigid-body constraints, not elastic material.
Heat shows relative recorded load strength; connecting lines explain which
force application points belong to the same rigid body, not internal stress.

Native execution retains every solver pose and records detailed force data at
up to 720 regular replay frames. The initial unsolved frame has no force claim.
To supplement an existing accepted run without rewriting its acceptance result:

```bash
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness.diagnostics .runtime_runs/projects/my_machine/runs/r01
```

Diagnostics save a new directory and publish only after successful physics and
original-geometry checks. Existing geometry audits are reusable only when all
solver timestamps and original body poses match exactly; otherwise the new
trajectory is audited. Add `--regions-only` to annotate an old pose record
without simulation or inferred forces. The dashboard URL parameter
`display=mechanics` opens motion replay directly.


### Coordinates in a moving part

`relative_position("bead", "tray", "x")` reads the bead COM in the tray's
COM-centered local frame, in metres. The recorded position and quaternion at
every solver sample determine this frame. Use corresponding COM-local region
bounds; original world-space bounds cannot be copied unchanged. `window` and
ordinary arithmetic retain the recorded time coordinates. This reads existing
motion only and adds no body force, constraint, damping or contact rule.

### Explicit solver iteration budget

`--physics-solver-iterations 320` records and applies a positive integer PSOR
iteration budget. The runtime request field is
`numerical_settings.solver_iterations`; it survives timestep retries and is
checked before cached results can be reused. The default remains 160. Native
friction-clutch cases retain their minimum of 240. `native_solver.max_iterations`
records the effective value. Changing this value does not relax geometry or
functional acceptance criteria.


### Controller inertia for fixed assemblies

Position/speed input stabilization uses the actual mass and inertia of the currently active fixed-joint components, including the parallel-axis contributions at the joint. A component attached to a fixed body has no relative inertial response. Broken fixed links are refreshed before control forces are calculated. Bodies, colliders, passive joint damping, target commands and effort limits remain unchanged. Component inertia is evaluated only when an active position/speed input needs it.
