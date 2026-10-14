# CAD toolchain

Agents working inside a FunCAD-Harness run read this file. It names the
interpreters and the tools that are available. Do not search for other Python
environments or install packages during a CAD task.

## Interpreters

The harness runs two separate environments (OCP and PyChrono ship independent
native libraries and must not share one Python):

| Purpose | Variable | Packages |
|---|---|---|
| CAD, geometry, STEP export, previews | `SIMREADY_CAD_PYTHON` | build123d 0.10, OCP 7.8, trimesh, coacd, python-fcl, scipy |
| Physics | `SIMREADY_PYCHRONO_EXECUTABLE` | PyChrono 10 |

The harness exports both variables to every agent turn. Use them directly:

```bash
"$SIMREADY_CAD_PYTHON" model.py            # build CAD with build123d
"$SIMREADY_CAD_PYTHON" -c "import build123d, OCP; print('ok')"
```

On Linux the CAD environment's `lib` directory is on `LD_LIBRARY_PATH`
(the OCP wheel needs the environment's `libexpat`). If an interpreter or its
imports fail, report `CAD_RUNTIME_UNAVAILABLE` and stop; environment discovery
is not part of a CAD task.

## Building, checking and simulating drafts

Use the harness author tools instead of ad-hoc scripts; they run under the same
limits as the final evaluation (see
`simulation_ready_cad_workflow/validation_pipeline/slim_harness/AUTHOR_TOOLS.md`):

```bash
H="$SIMREADY_CAD_PYTHON -m simulation_ready_cad_workflow.validation_pipeline.slim_harness --workspace <repository root>"
$H build model.py
$H preview scene.json
$H check scene.json
$H simulate scene.json --test <id>
```

## Rendering

| Path | Status |
|---|---|
| Harness preview (`slim_harness preview`) and the dashboard's original-geometry viewer | available |
| Generated `.step.glb` sidecar files | available |
| Deterministic STEP/OCP inspection | available (geometry validation, not images) |
| Blender, FreeCAD, OpenSCAD, MeshLab | not provided |

Do not invoke a renderer that is not listed as available. State
`visual render unavailable` once and continue with deterministic inspection.

## Assembly interpretation

The imported STEP structure owns the verdict: one solid may be a part; several
uniquely named children under a labelled Compound are an assembly. Render
motion and failures from the original STEP solids, never from collider proxies.

## Output policy

- Write sources and artifacts in the design directory you were given; runtime
  outputs belong under `.runtime_runs/`.
- Run the concise deterministic checks the task needs and return paths and
  results. Do not compute hashes or write standalone reports unless asked.
- The harness owns physical results and certificates; never edit a PASS label
  or relax a task to force success.
