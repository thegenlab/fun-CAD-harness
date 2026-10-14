"""Resource bounds for *building* benchmarks, never frozen evaluation rules.

These ceilings ask the author to replan; they do not turn an incomplete test
into a physical PASS. Slim currently uses original triangle meshes, not convex
decomposition. Count original solids/instances and any explicit decomposition
separately from triangles so a mesh is not misleadingly called one cheap unit.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping

from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import BoundedSubprocessError, run_bounded_subprocess


# A simulation still running at its configured limit is not restarted: it keeps
# running to twice the limit and then to this ceiling, its hard wall limit.
SIMULATION_EXTENSION_CEILING_SECONDS = 1800.0


def simulation_time_steps(configured):
    """The configured limit and its extensions, e.g. 300 -> (300, 600, 1800)."""
    configured = float(configured)
    ceiling = max(configured, SIMULATION_EXTENSION_CEILING_SECONDS)
    return tuple(sorted({configured, min(2 * configured, ceiling), ceiling}))


def simulation_extension(configured, elapsed):
    """Record which extension a finished or stopped simulation reached."""
    steps = simulation_time_steps(configured)
    reached = next((step for step in steps if elapsed <= step), steps[-1])
    return {"configured_timeout_seconds": steps[0], "timeout_steps_seconds": list(steps),
            "extended_to_seconds": reached if reached > steps[0] else None}


def simulation_activity(project):
    """Mark simulation wall time; it excludes Harness-side waiting, not concurrent author work."""
    from .construction_budget import LEDGER_NAME, ProjectBudget
    if project is None or not (Path(project) / LEDGER_NAME).is_file():
        return nullcontext()
    return ProjectBudget(project).activity("simulation")


REVISION_STRATEGIES = [
    {"id": "modify_mechanism", "instruction": "Modify mechanical design when evidence supports a design issue."},
    {"id": "simplify_simulation", "instruction": "Benchmark construction only: use flexible / abstract joints, finite equivalent elasticity, or static original-geometry checks plus abstract joints; simplify nonfunctional geometry while retaining core motion, load and interference evidence. Declare the reduced validation scope."},
    {"id": "relax_noncore_task", "instruction": "Benchmark construction only: reduce secondary loads, repetitions or precision without changing the task's core semantics; record the change and its reason. Never rewrite a frozen evaluation task or report an unrun test as passed."},
]


class ConstructionLimitError(ValueError):
    def __init__(self, code: str, message: str, **details):
        self.code = code
        self.details = {"code": code, "message": message, **details,
                        "revision_strategies": REVISION_STRATEGIES}
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ConstructionLimits:
    simulation_timeout_seconds: float = 180.0
    max_tasks: int = 3
    max_tests: int = 3
    # Practical inventory ceilings, not a claim that 2048 solids finish in 3min.
    max_geometry_units: int = 2048
    max_geometry_units_per_body: int = 128
    max_geometry_triangles: int | None = None
    max_geometry_triangles_per_body: int | None = None
    audit_timeout_seconds: float = 600.0

    @classmethod
    def from_policy(cls, policy):
        value = asdict(policy) if is_dataclass(policy) else dict(policy)
        value = value.get("config", value)
        result = cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})
        for key, limit in asdict(result).items():
            if limit is None and key.startswith("max_geometry_triangles"):
                continue
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
                raise ValueError(f"{key} must be a positive finite number")
            if key not in {"simulation_timeout_seconds", "audit_timeout_seconds"} and int(limit) != limit:
                raise ValueError(f"{key} must be an integer")
        return result


def check_task_test_counts(scene: Mapping[str, Any], policy, *, task_count: int | None = None):
    limits = ConstructionLimits.from_policy(policy)
    count = len(scene.get("tests", []))
    if task_count is None and isinstance(scene.get("tasks"), list):
        task_count = len(scene["tasks"])
    if count > limits.max_tests or (task_count is not None and task_count > limits.max_tasks):
        raise ConstructionLimitError("CONSTRUCTION_TASK_TEST_LIMIT",
            "Choose a compact set of tests covering the core mechanical requirements before running geometry or physics.",
            test_count=count, max_tests=limits.max_tests, task_count=task_count, max_tasks=limits.max_tasks)
    return {"test_count": count, "task_count": task_count}


def check_geometry_counts(unit_counts: Mapping[str, int], policy, *, triangle_counts=None):
    """Use during inventory/decomposition as well as before scene construction.

    A decomposition producer can call this after each emitted component to stop
    at the cap; no expensive union/export of an already over-budget set is needed.
    """
    limits = ConstructionLimits.from_policy(policy)
    counts = dict(unit_counts)
    triangles = dict(triangle_counts or {})
    total = sum(counts.values())
    too_many = {body: count for body, count in counts.items() if count > limits.max_geometry_units_per_body}
    triangle_excess = {body: count for body, count in triangles.items()
                       if limits.max_geometry_triangles_per_body is not None and count > limits.max_geometry_triangles_per_body}
    triangle_total = sum(triangles.values())
    report = {"geometry_unit_count": total, "geometry_units_by_body": counts,
              "triangle_count": triangle_total, "triangles_by_body": triangles,
              "max_geometry_units": limits.max_geometry_units,
              "max_geometry_units_per_body": limits.max_geometry_units_per_body}
    if (total > limits.max_geometry_units or too_many or triangle_excess
            or (limits.max_geometry_triangles is not None and triangle_total > limits.max_geometry_triangles)):
        raise ConstructionLimitError("CONSTRUCTION_GEOMETRY_LIMIT",
            "Geometry exceeds this construction project's resource cap. Simplify the simulation representation or use static original-geometry checks plus flexible / abstract joints before retrying.",
            **report, over_limit_bodies=too_many, triangle_over_limit_bodies=triangle_excess,
            max_geometry_triangles=limits.max_geometry_triangles)
    return report


def scene_preflight(scene, policy):
    counts = check_task_test_counts(scene, policy)
    # Explicit assignments can reject an excessive inventory before STEP import.
    # Actual source inventory is checked again; this is only a cheap lower bound.
    units = {body["id"]: len(body["solid_ids"]) if isinstance(body.get("solid_ids"), list) else 1
             for body in scene.get("bodies", []) if isinstance(body, dict) and "id" in body}
    return {**counts, **check_geometry_counts(units, policy)}


def geometry_preflight(geometry, policy):
    units, triangles = {}, {}
    for body in geometry.get("bodies", []):
        source_count = len(body.get("source_components", body.get("source_solid_ids", []))) or 1
        # New producers may explicitly supply convex pieces. Account for them
        # instead of counting their parent mesh as one item.
        component_count = len(body.get("collision_components", []))
        units[body["id"]] = max(source_count, component_count)
        triangles[body["id"]] = int(body.get("triangle_count", body.get("fidelity", {}).get("triangle_count", 0)))
    return check_geometry_counts(units, policy, triangle_counts=triangles)


def workload_key(scene, geometry, test):
    """Timing records are reusable only for the identical authored workload."""
    location_fields = {"source_file", "mesh_file", "audit_mesh_file", "model_file", "scene_file", "source_stat"}
    def normalized(value):
        if isinstance(value, dict):
            return {key: (sorted(item.values()) if key == "source_bindings" and isinstance(item, dict)
                          else normalized(item)) for key, item in value.items() if key not in location_fields}
        if isinstance(value, list):
            return [normalized(item) for item in value]
        return value
    sources = [{key: body.get(key) for key in ("id", "source_sha256", "mesh_sha256")}
               for body in geometry.get("bodies", [])]
    from .implementation import native_implementation_binding
    value = {"scene": normalized(scene), "sources": sources, "test": normalized(test),
             "native_implementation": native_implementation_binding()}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def simulation_preflight(scene, geometry, test, policy, *, remaining_seconds=None, measurements=()):
    limits = ConstructionLimits.from_policy(policy)
    report = {**scene_preflight(scene, limits), **geometry_preflight(geometry, limits)}
    time_steps = simulation_time_steps(limits.simulation_timeout_seconds)
    # Simulation time is not charged, so the remaining budget does not shorten a run.
    timeout = time_steps[-1]
    if remaining_seconds is not None:
        if not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
            raise ConstructionLimitError("PROJECT_BUDGET_EXHAUSTED", "Reserve the remaining time for the completion summary.")
    # Match numerical.run_fixed_attempts: authored tests may ask for a finer
    # step, while the native starting step never exceeds 1ms.
    step = min(.001, float(test.get("step_s", .001)))
    duration = float(test.get("duration_s", 0.0))
    steps = math.ceil(duration / step) if step > 0 else None
    key = workload_key(scene, geometry, test)
    report.update(effective_timeout_seconds=timeout, configured_timeout_seconds=time_steps[0],
                  timeout_steps_seconds=list(time_steps), workload_key=key, test_id=test.get("id"),
                  requested_simulated_seconds=duration, estimated_step_count=steps,
                  estimate_kind="inventory_only; no reliable wall-time prediction",
                  revision_strategies=REVISION_STRATEGIES)
    for measurement in measurements:
        if measurement.get("workload_key") != key:
            continue
        if (measurement.get("status") == "timeout"
                and float(measurement.get("effective_timeout_seconds", 0)) >= timeout):
            raise ConstructionLimitError("SIMULATION_UNCHANGED_TIMEOUT",
                "This unchanged workload already exhausted an equal or larger timeout. Repeating it is disallowed by the construction budget policy; change the mechanism, simulation representation or noncore test before retrying. This is a retry policy, not proof of a unique physical cause.",
                previous_elapsed_seconds=measurement.get("elapsed_seconds"), **report)
        # An elapsed measurement is not a proof that every future run is slower.
        # Only a caller-supplied lower bound with an explicit basis rejects early.
        lower = measurement.get("proven_minimum_seconds")
        if lower is not None and measurement.get("lower_bound_basis") and float(lower) > timeout:
            raise ConstructionLimitError("SIMULATION_COMPLEXITY_EXCEEDS_TIMEOUT",
                "The supplied workload lower bound exceeds the available simulation time; select a construction simplification first.",
                proven_minimum_seconds=lower, lower_bound_basis=measurement["lower_bound_basis"], **report)
        measured = measurement.get("elapsed_seconds")
        if measured is not None:
            report.update(estimated_wall_seconds=float(measured), estimate_kind="same_workload_measurement_not_guarantee",
                          simplification_recommended=float(measured) >= time_steps[0])
    return report


def run_simulation_bounded(command, *, scene, geometry, test, policy, remaining_seconds=None,
                           measurements=(), cwd=None, env=None, max_rss_bytes="auto",
                           report_path: Path | None = None, history_dir: Path | None = None):
    """Bound loading, solving, native retries and descendants with one deadline."""
    if history_dir is None and os.environ.get("SIMREADY_CONSTRUCTION_PROJECT"):
        history_dir = Path(os.environ["SIMREADY_CONSTRUCTION_PROJECT"]) / "construction-simulations"
    history_file = Path(history_dir) / (workload_key(scene, geometry, test) + ".json") if history_dir else None
    if max_rss_bytes=="auto" and history_dir is not None:
        from .physics_invocation import _rss_limit
        max_rss_bytes=_rss_limit(Path(history_dir).parent,None)
    previous = []
    if history_file is not None and history_file.is_file():
        previous = [json.loads(history_file.read_text(encoding="utf-8"))]
    try:
        report = simulation_preflight(scene, geometry, test, policy,
                                     remaining_seconds=remaining_seconds, measurements=[*previous, *measurements])
    except ConstructionLimitError as exc:
        if report_path is not None:
            write_json_atomic(Path(report_path), {"status": "preflight_replan", "elapsed_seconds": 0., **exc.details})
        raise
    started = time.monotonic()
    report["status"] = "running"
    try:
        child_env = dict(os.environ if env is None else env)
        if "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime" in command:
            # The runtime CLI also bounds direct author invocations. Mark this
            # exact request so a supervisor-owned bound does not recursively
            # launch another supervisor or reset the timeout on native retries.
            child_env["SIMREADY_CONSTRUCTION_GUARDED_REQUEST"] = str(Path(command[-1]).resolve())
        from .execution import geometry_inputs,run_compute
        project = Path(history_dir).resolve().parent if history_dir is not None else None
        request = Path(command[-1]).resolve()
        with simulation_activity(project):
            result = run_compute(command, cwd=cwd, env=child_env,
                workspace=Path(cwd).resolve() if cwd is not None else Path(__file__).resolve().parents[3],
                project=project,inputs=[request,*geometry_inputs(geometry)],outputs=[request.parent],
                timeout_seconds=report["effective_timeout_seconds"], max_rss_bytes=max_rss_bytes,
                runner=run_bounded_subprocess)
        report.update(status="finished" if result.returncode == 0 else "process_failed", returncode=result.returncode,memory_budget=getattr(result,"memory_budget",None))
        return result
    except BoundedSubprocessError as exc:
        report.update(status="timeout" if exc.code == "BOUNDED_SUBPROCESS_TIMEOUT" else "resource_error",
                      subprocess_error=exc.code,memory_budget=getattr(exc,"memory_budget",None))
        if exc.code == "BOUNDED_SUBPROCESS_TIMEOUT":
            extended = (f", extended from its {report['configured_timeout_seconds']:g}s limit"
                        if report["effective_timeout_seconds"] > report["configured_timeout_seconds"] else "")
            raise ConstructionLimitError("SIMULATION_HARD_TIMEOUT",
                f"The simulation process tree was stopped at its {report['effective_timeout_seconds']:g}s wall limit{extended}. "
                "This does not uniquely diagnose a mechanical defect; choose a revision strategy using the available evidence.",
                test_id=test.get("id"), effective_timeout_seconds=report["effective_timeout_seconds"],
                configured_timeout_seconds=report["configured_timeout_seconds"]) from exc
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report.update(simulation_extension(report["configured_timeout_seconds"], report["elapsed_seconds"]))
        if report_path is not None:
            write_json_atomic(Path(report_path), report)
        if history_file is not None:
            write_json_atomic(history_file, report)


def run_runtime_entrypoint(request_path: Path, project: Path, *, emit_output=True, raise_on_failure=False):
    """Apply the same construction bounds to a direct ``-m runtime`` command."""
    from .construction_budget import ProjectBudget, BudgetExhausted
    from .frozen_experiment import load_experiment

    request_path, project = Path(request_path).resolve(), Path(project).resolve()
    budget = ProjectBudget(project)
    if load_experiment(project) is not None:
        raise ValueError("Construction policy cannot run a frozen evaluation experiment")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = request_path.parent / "construction-runtime-result.json"
    result = {"schema_version": "slim_author_tool_result_v1", "command": "direct_runtime", "project": str(project),
              "certified": False, "construction_only": True, "tests": [], "revision_strategies": REVISION_STRATEGIES,
              "scope": "direct construction trial; native trace only, original-geometry audit and review not completed"}
    started = time.monotonic()
    try:
        with budget.activity("simulation"):
            snapshot = budget.check_available()
            child = run_simulation_bounded([sys.executable, "-X", "utf8", "-m",
                "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime", str(request_path)],
                scene=request["scene"], geometry=request["scene"], test=request["test"], policy=snapshot["config"],
                remaining_seconds=snapshot["work_remaining_seconds"], cwd=Path.cwd(),
                history_dir=project / "construction-simulations", report_path=request_path.parent / "construction-cost.json")
            result.update(status="DEVELOPMENT_RUNTIME_FINISHED" if child.returncode == 0 else "DEVELOPMENT_FAILURE",
                          returncode=child.returncode, trace_file=request.get("output"))
            (request_path.parent / "construction-runtime.stdout.log").write_bytes(child.stdout)
            (request_path.parent / "construction-runtime.stderr.log").write_bytes(child.stderr)
            if emit_output:
                sys.stdout.buffer.write(child.stdout)
                sys.stderr.buffer.write(child.stderr)
            if child.returncode and raise_on_failure:
                raise RuntimeError(f"Native construction simulation failed; inspect {request_path.parent / 'construction-runtime.stderr.log'}")
            return child.returncode
    except (ConstructionLimitError, BudgetExhausted) as exc:
        result.update(status="BUDGET_EXHAUSTED" if isinstance(exc, BudgetExhausted) else "NEEDS_SIMPLIFICATION",
                      issues=[exc.details], required_action="write_completion_summary" if isinstance(exc, BudgetExhausted) else "choose_revision_strategy")
        if raise_on_failure:
            raise
        print(json.dumps(result), file=sys.stderr)
        return 2
    finally:
        result.update(elapsed_seconds=time.monotonic() - started, construction_budget=budget.snapshot())
        write_json_atomic(result_path, result)
        # Direct requests contain compiled metre-space scenes, not necessarily
        # the submitted source scene. Do not invent a matching source binding or
        # count such a request as an accepted test in the completion report.
        key = hashlib.sha256(str(request_path).encode()).hexdigest()[:20]
        write_json_atomic(project / "construction-evidence" / f"direct-{key}.json", {
            "schema_version": "construction_evidence_v1", "project": str(project),
            "result_file": str(result_path), "source": str(request_path), "source_scene_sha256": None,
            "command": "direct_runtime", "status": result.get("status"), "test_ids": [request["test"].get("id")],
        })


def run_imported_simulation(prepared, test, output, project):
    """Public Python API keeps its dict return type while native work is bounded.

    Preserve each request beside the requested trace. The child CLI enters only
    its exact native call, so this dispatch does not recursively spawn children.
    """
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=output.stem + ".construction-", dir=output.parent))
    request = folder / "request.json"
    write_json_atomic(request, {"scene": prepared, "test": test, "output": str(output)})
    run_runtime_entrypoint(request, project, emit_output=False, raise_on_failure=True)
    return json.loads(output.read_text(encoding="utf-8"))
