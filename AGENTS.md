# Agent instructions

Coding agents working in this repository — contributors' assistants and the
author/reviewer agents of a harness run — follow these rules.

## Running experiments for a user

If a user asks you to run an experiment from the repository root, follow
"Running an experiment from a coding agent" in [README.md](README.md). Author
and reviewer agents started by the harness work inside `.runtime_runs/` and
never start, stop or resume runs.

## CAD and runtime

- Read [CAD_TOOLCHAIN.md](CAD_TOOLCHAIN.md) before CAD creation, modification,
  inspection or rendering. It names the interpreters and available renderers.
- Runtime outputs belong under `.runtime_runs/`. Render motion and failures from
  the original STEP solids, never from collider proxies.
- A named multi-child STEP compound is an assembly. Keep valid geometry if a
  helper reports `kind=part`; report the helper mismatch instead.

## Harness results

- The harness owns physical results and certificates. Never edit a PASS label,
  a result file or the harness to force success, and never relax a task without
  recording it through the documented adjustment files.
- Preserve completed models, tests and evidence; keep a passing design as it is.
- When a mechanical case needs a new revision and simulation cost is high,
  consider the harness's source-bound flexible joints and equivalent contact
  models (`simulation_ready_cad_workflow/validation_pipeline/slim_harness/EQUIVALENT_ELASTICITY.md`),
  keeping printable source geometry and the functional targets.

## Development

Run the affected tests from the repository root with the CAD environment's Python:

```bash
python -m unittest simulation_ready_cad_workflow.validation_pipeline.tests.<module>
```

Keep changes small and in the style of the surrounding code.
