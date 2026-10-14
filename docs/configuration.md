# Configuration

## Environment variables

| Variable | Meaning |
|---|---|
| `SIMREADY_CAD_PYTHON` | Interpreter of the CAD environment (build123d, OCP, trimesh …). Runs the harness, the dashboard, CAD builds and previews. |
| `SIMREADY_PYCHRONO_EXECUTABLE` | Interpreter of the PyChrono environment. A project can override it with `create --chrono-python`. |
| `LD_LIBRARY_PATH` | Linux only: the CAD environment's `lib` directory first (the OCP wheel needs its `libexpat`). |
| `PYTHONNOUSERSITE=1` | Keeps packages from `~/.local` out of both environments. |
| `SIMREADY_CONTAINER_ISOLATION=1` | Set in the Docker image: the container is the isolation boundary, so agent CLIs run without their own nested sandbox. |
| `SIMREADY_RENDER_LANG` | Language of labels baked into rendered images (`en` or `zh`). |
| `SIMREADY_TEST_CHRONO_PYTHON` | Enables the PyChrono integration tests. |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Optional API-key authentication for Codex CLI / Claude Code instead of their own logins. |

`scripts/setup_conda.sh` writes the first four and `SIMREADY_RENDER_LANG=en` to `.env`; `scripts/funcad-gui`
loads `.env`. The Docker image sets them itself.

## Modes

`create --mode free --task task.md` (or `--brief "text"`) starts a new
mechanism. The author agent writes the task template; the independent reviewer
reviews it and binds it to the design. Free creation is the only mode in this
release; `--mode` may be omitted.

## Budgets

| Tier (`--tier`) | Total time (recommended) | Per simulation (recommended) | Tests per design | Parts |
|---|---|---|---|---|
| `light` | 1 h | 1 min | 2 | 10 |
| `medium` | 2 h | 3 min | 2 | 15 |
| `high` (default) | 5 h | 10 min | 5 | 20 |

The total time counts active work of agents, CAD and audits, not waiting for
a provider. Each tier comes with recommended limits; choose your own
with `--budget-hours {1,2,3,5,8,12}` and `--simulation-timeout SECONDS` (the GUI
offers the same hours and 1–20 minutes per simulation). A simulation that runs
longer than its limit is stopped and recorded as a timeout; `--audit-timeout`
limits the source-motion audit. In construction runs (`--construction`: every
GUI run and the `high` tier default) simulation time does not count toward the
total, and a simulation still running at its limit is not stopped but keeps
running — up to twice the limit, capped at a 30-minute ceiling; only a
simulation unfinished then is recorded as a timeout. When the total time is
used up the run finishes with the best evidence it has; `resume` continues a
project with its saved settings.

```bash
python -m simulation_ready_cad_workflow.validation_pipeline.slim_harness budget status --project cam_toy
```

## Agents and models

| `--worker-provider` | Default model | Reasoning flag |
|---|---|---|
| `codex` | `gpt-6-astra` | `--reasoning low…xhigh` (default `high`) |
| `claude_code` | `claude-opus-5-5` | `--effort` (`ultra` maps to `max`) |
| `cursor` | Cursor default | not supported |
| `kimi` | `kimi-code/k3` | not supported |

Use `--model` to pick another model of the same provider. The reviewer is chosen
independently with `--reviewer-provider`, `--reviewer-model`,
`--reviewer-reasoning` and `--reviewer-executable`; by default Codex and Cursor
authors get a Codex reviewer at `medium` reasoning and other authors a reviewer
of their own agent and model. Login, quota and connection failures are
recorded as provider errors and are never returned to the author as design
defects.

## Dashboard

```bash
python -m simulation_ready_cad_workflow.validation_pipeline.dashboard serve \
  [--host 127.0.0.1] [--port 8765] [--enable-launch] [--max-launches 4] [--base-path /prefix]
```

Without `--enable-launch` the dashboard is a read-only viewer of
`.runtime_runs/`; the Run monitor (`/runs`) still shows every run. With it, the
New task page, the launch queue and the Stop/Resume buttons are enabled; their
requests need the per-process token printed at startup (`#human=…` in the
link). `--max-launches` limits how many runs started from the dashboard may run
at the same time. Queued runs start one at a time, in order, once no
dashboard-started run is active; the queue is stored in
`.runtime_runs/dashboard_launches/queue.json` and only advances while a
launch-enabled dashboard is serving. Runs keep going if the dashboard stops. Binding to a non-loopback address additionally
requires `--allow-remote-launch`; do not expose a launch-enabled dashboard to an
untrusted network.

## Output layout

```
.runtime_runs/projects/<project>/
  task.md, workflow.json, mode.json       task and workflow state
  worker_turns/tNNN_<kind>/               prompts, events and responses of each agent turn
  runs/rNN/design/                        the revision's CAD source, STEP, scene.json, task_template.py
  runs/rNN/template_runs/aNNN/            prepared model, reviewer binding, trial results
  runs/rNN/result.json                    verdict and evidence of the revision
  completion_summary.md                   final summary
```
