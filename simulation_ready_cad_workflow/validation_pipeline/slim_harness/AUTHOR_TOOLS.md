# Continuous authoring in Slim

The existing author session can build, preview, check and try its design without
ending the turn or opening another revision. These tools use the existing Slim
geometry and physics, not Asset Builder's legacy IR/backend. Final acceptance
still uses normal `create` / `submit` / `resume`, program checks and every
required test. Model review only advises on changes to the task or `scene.tests`;
see [Review and evidence ownership](README.md#review-and-evidence-ownership).
A draft result never creates a mechanical PASS or certificate.

File search and reads stay inside the current project directory. Do not list
or open other project folders under `.runtime_runs/projects/`, and do not read
another machine's `model.py`, `scene.json`, construction plan, or `design/`.

Run from the repository with the CAD environment's Python. The examples below use
`python` as shorthand for that interpreter (`$SIMREADY_CAD_PYTHON` when it is set).
Put global `--workspace` before the command when running from a design directory.

```text
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness build model.py
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness preview model.py
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness preview scene.json
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness check scene.json
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness probe scene.json --test loaded_motion
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness simulate scene.json --test loaded_motion
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness audit path/to/request.json
```

- `build`: generate STEP bodies and `scene.json` next to the Python source (or
  under an explicit `--output`). A preserved build attempt is kept before copying
  the generated files into the design. No formal revision or verdict is created.
- `preview`: source geometry and four original-shape views; no tests required.
- `check`: validate the actual scene, joints and initial source geometry. It does
  not call an LLM or prove the machine works.
- `simulate`: run one complete named test and retain its native trace. When the
  scene has several tests, select one with `--test`; the tool will list choices
  instead of starting an entire validation sequence inside an author turn.
  Native physics feedback returns first. Saved-motion audit is marked
  `deferred_to_formal_validation`; the complete formal checks still run before
  acceptance. Use `audit` explicitly when motion geometry helps diagnose a failure.
- `probe`: construction-only cost measurement, capped at 15 wall seconds and
  normally 0.1 simulated seconds. Registered measurement tests retain their exact
  original definition and run only until the wall deadline. A confirmed prefix
  can supply progress after timeout. The estimate includes startup and may miss
  later contacts; it is not a reliable lower bound. Probe assertions do not count,
  no physics cache is written, and no first-pass coverage credit is given.
  Read `cost_probe.required_action`, then revise with evidence or run the full test.
- `audit`: run the same source-bound trajectory audit on an existing simulation
  request and its trace, without rerunning physics. `--trace` can select a saved
  numerical-attempt trace or confirmed `prefix.json`; `--output` selects a new
  report path. A prefix audit covers only its confirmed rows and grants no PASS.
  The report gives original/selected/checked counts and selected source indices;
  unsampled states and continuous clearance are not claimed. Initial and
  prescribed-pose callers retain their full supplied pose list.

The optional `--project` attaches an external draft to an existing Slim project.
Drafts inside a project's `runs/*/design` automatically use that project's frozen
experiment, when present. `build --output` also checks the destination project
before writing; a project destination must be a run's design directory.
A frozen test's loads, duration, units, gravity and
assertions cannot be edited or shortened. A test can be selected for development;
the formal workflow still requires the complete set. Existing registered
measurements are checked against the materialized snapshot, including the
registered source-plan files and injected frozen tests.

Once source and tests are ready, submit normally; do not keep the author turn
open waiting for final acceptance. Formal validation can reuse completed
development physics only when the compiled
inputs, actual source/mesh bytes, test, registered measurement, selected Chrono
interpreter and calculation implementation match. Changed or incomplete records
cause normal execution. The required source-motion audit and program gates
still run. Path-dependent model scopes must match exactly; old scopes are not
silently re-signed to manufacture reuse. Reused traces retain their original
physics time and separately report the actual reuse time.

Every command gets a separate directory under `.runtime_runs/author_tools/`,
including its source snapshot, reports, logs and total elapsed seconds. Python
source is saved before execution; declared dependencies are saved once `build()`
returns its Asset, before validating/exporting its parts. Sources
and local helpers must be under the scene/model directory. Preview/check/simulate
do not modify the live design, formal attempts or published results.
Do not count a development check as an independent benchmark attempt; include its
cost in the enclosing author session/experiment cost.

## Named parts instead of solid numbers

This optional source interface is adapted from the earlier Asset Builder prototype's `Asset` / `Part` idea.
Existing hand-authored Slim scenes remain valid. Define `build()` in `model.py`:

```python
from build123d import Box, Pos
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.asset import Asset, Part

def build():
    # Geometry-only draft: previewable, not a complete mechanical test.
    return Asset(parts={
        "base": Part(Box(40, 30, 4), body="frame", role="fixed"),
        "moving_piece": Part(Pos(0, 0, 8) * Box(10, 10, 4), body="carriage"),
    }, description="Named draft parts")
```

Use stable part and body names. Multiple parts may belong to one rigid body,
with the same fixed/dynamic role and density. The tool exports one STEP per body
and a `parts.json` inventory; you do not write `solid_N` references. This grouping
does not prove physical attachment: the author must justify it from the source,
and program source checks remain required.

Add `Asset.joints`, `Asset.tests` and `Asset.gravity_m_s2` using the normal Slim
fields from README. `Part.color` accepts the usual numeric RGB/RGBA list;
`Part.density_kg_m3` defaults to 7800. CAD coordinates are always millimetres;
`Asset.units` selects `mm` (default) or `m` for scene/joint coordinates. Motion,
force and assertion signals keep the normal SI units. An existing equivalent
elasticity declaration may be supplied explicitly as `Asset.equivalent_elasticity`.

List local helper/data files in `Asset.dependencies`; these are copied with the
model and included in authoring sources. Preserve all actual physical parts;
there is no display-only role that can conceal a supporting or contacting part.

With a frozen benchmark, keep its required body/joint IDs and omit `Asset.tests`
to use the frozen tests. With ordinary authoring, the initial task and first test
set become the baseline without model review. Later task/test changes receive
one advisory review; geometry-only changes do not. The author decides how to use
the advice, while program gates remain enforced. Return `ready`
only after `build` has produced the normal final `design/scene.json` delivery.

## First pass before repeated development trials

For an attached project, `simulate` first checks its declared test set. Once a test definition has received an execution attempt, repeating it while another definition is untested returns `FIRST_PASS_PENDING`, `required_action: run_unvisited_test` and the remaining IDs. This check precedes expensive geometric preparation. Complete the other first trials before repeatedly repairing one test. The coverage ledger does not grant acceptance; formal validation still requires current source-bound physics, assertions and geometry checks.
