"""Direct draft tools inside the existing Slim author session.

Adapt the Asset Builder prototype's preview/check/simulate separation without importing its legacy
package compiler or a second agent loop. Draft attempts cannot publish PASS.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import run_bounded_subprocess, BoundedSubprocessError
from ..trajectory.capture import publish_attempt
from ..trajectory.configuration import model_binding
from .budget import ActiveBudgetExhausted
from .contracts import load_scene, compile_scene
from .frozen_experiment import load_experiment, bind_scene
from .geometry import prepare_geometry
from .audit import audit_motion, audit_saved_motion
from .evidence import render_views
from .trace_storage import open_trace, release_trace_cache
from .measurement_registry import load_registered_measurements, validate_trace_measurements
from .workflow import CHRONO_PYTHON
from .construction import construction_requested
from .construction_budget import BudgetExhausted, ProjectBudget, LEDGER_NAME
from .construction_limits import (ConstructionLimitError, REVISION_STRATEGIES,
                                 scene_preflight, run_simulation_bounded, simulation_activity, simulation_time_steps)


def project_for_scene(scene: Path, workspace: Path, project=None):
    base = workspace / ".runtime_runs" / "projects"
    inferred = next((p for p in scene.parents if p.parent == base), None)
    explicit = None
    if project is not None:
        explicit = Path(project)
        if not explicit.is_absolute(): explicit = base / explicit
        explicit = explicit.resolve()
        if explicit.parent != base or not (explicit / "workflow.json").is_file():
            raise ValueError("--project must name an existing Slim project")
    if inferred and explicit and inferred != explicit:
        raise ValueError("Scene belongs to a different Slim project")
    return inferred or explicit


def copy_sources(source_root, destination, references):
    for relative in references:
        path = (source_root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(f"Keep draft source dependencies below its scene directory: {relative}")
        target = destination / relative
        if target.is_file() and target.read_bytes()!=path.read_bytes():
            raise ValueError(f"Source dependency conflicts with generated snapshot: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def snapshot_scene(source: Path, destination: Path, experiment, extra_sources=()):
    # Preserve even a malformed or changed draft before validation rejects it.
    shutil.copy2(source,destination/"submitted-scene.json")
    scene = bind_scene(json.loads(source.read_text(encoding="utf-8-sig")), experiment)
    references = {b["source"] for b in scene["bodies"]} | set(scene.get("authoring_sources", [])) | set(extra_sources)
    copy_sources(source.parent,destination,references)
    write_json_atomic(destination / "scene.json", scene)
    return destination / "scene.json"


def _build(source, destination, attempt, workspace, *, timeout_seconds=300, project=None, cad_python=None):
    shutil.copy2(source,attempt/source.name)
    result_file = attempt / "build-result.json"
    from .execution import run_compute
    result = run_compute([str(cad_python or sys.executable), "-X", "utf8", "-m",
        "simulation_ready_cad_workflow.validation_pipeline.slim_harness.asset",
        str(source), str(destination), str(result_file)], cwd=workspace,
        workspace=workspace, project=project, inputs=[source.parent], outputs=[attempt],
        timeout_seconds=timeout_seconds, max_rss_bytes="auto", runner=run_bounded_subprocess)
    (attempt / "build.stdout.log").write_bytes(result.stdout)
    (attempt / "build.stderr.log").write_bytes(result.stderr)
    if result.returncode or not result_file.is_file():
        raise ValueError(f"Asset build failed; inspect {attempt / 'build.stderr.log'}")
    return Path(json.loads(result_file.read_text(encoding="utf-8"))["scene"])


def run_tool(command, source, *, workspace, project=None, output=None, test_id=None, chrono_python=CHRONO_PYTHON, cad_python=None, visual_policy='full', visual_policy_log=None):
    from ..benchmark_sandbox.visual_policy import WORKER_NO_VISUALS, deny, effective_policy
    visual_policy = effective_policy(visual_policy)
    if visual_policy == WORKER_NO_VISUALS and command == 'preview':
        deny('preview', visual_policy_log)
    source, workspace = Path(source).resolve(), Path(workspace).resolve()
    root = workspace / ".runtime_runs" / "author_tools"
    root.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{command}-", dir=root))
    started = time.monotonic()
    result = {"schema_version": "slim_author_tool_result_v1", "command": command,
              "source": str(source), "attempt": str(attempt), "certified": False,
              "scope": "development only; final Slim review and all approved tests remain required"}
    activities = ExitStack()
    selected_project = construction_budget = limits = None
    active_definition=None
    try:
        if command not in {"build", "preview", "check", "simulate", "probe"}:raise ValueError("Unknown draft command")
        inherited_project = os.environ.get("SIMREADY_CONSTRUCTION_PROJECT")
        selected_project = project_for_scene(source, workspace, project or inherited_project or None)
        destination = (Path(output).resolve() if output else source.parent) if command=="build" else None
        if destination is not None:
            target_project=project_for_scene(destination/"scene.json",workspace)
            if target_project:
                if selected_project and selected_project!=target_project:raise ValueError("Build source and destination belong to different projects")
                relative=destination.relative_to(target_project).parts
                if len(relative)<3 or relative[0]!="runs" or relative[2]!="design":
                    raise ValueError("Build outputs inside a Slim project must stay in its run design directory")
                selected_project=target_project
        experiment = load_experiment(selected_project) if selected_project else None
        if selected_project and (selected_project / LEDGER_NAME).is_file() and construction_requested(selected_project):
            if experiment is not None:
                raise ValueError("Construction simplification policy cannot be applied to a frozen evaluation experiment")
            construction_budget = ProjectBudget(selected_project)
            limits = construction_budget.check_available()["config"]
            activities.enter_context(construction_budget.activity(command))
            result.update(project=str(selected_project), construction_only=True,
                          scope="benchmark construction; development evidence only, no frozen-evaluation certificate",
                          revision_strategies=REVISION_STRATEGIES)
        if command=="probe" and not construction_budget:
            raise ValueError("Cost probes belong to construction projects; frozen/evaluation tests stay unchanged")
        extra_sources=set()
        if selected_project and (selected_project/"measurement-registration.json").is_file():
            registration_path=selected_project/"measurement-registration.json"
            registry=json.loads(registration_path.read_text(encoding="utf-8"))
            extra_sources={p for b in registry["bindings"] for p in b["source_plan_bindings"]}
            shutil.copy2(registration_path,attempt/registration_path.name)
        result.update(project=str(selected_project) if selected_project else None)
        publish_attempt(workspace,attempt,result,project=selected_project,destination=destination)
        snapshot = attempt / "design"; snapshot.mkdir()
        if source.suffix.lower() == ".py":
            build_limits = {"timeout_seconds": min(300, construction_budget.check_available()["work_remaining_seconds"])} if construction_budget else {}
            if cad_python is not None:build_limits['cad_python'] = cad_python
            scene_path = _build(source, snapshot, attempt, workspace, project=selected_project, **build_limits)
            materialized = bind_scene(json.loads(scene_path.read_text(encoding="utf-8")), experiment)
            write_json_atomic(scene_path, materialized)
            copy_sources(source.parent,snapshot,extra_sources)
        else:
            if command == "build": raise ValueError("build requires model.py with build() returning Asset")
            scene_path = snapshot_scene(source, snapshot, experiment, extra_sources)
        if limits is not None:
            scene_preflight(json.loads(scene_path.read_text(encoding="utf-8-sig")), limits)
            construction_budget.check_available()
        result.update(scene_snapshot=str(scene_path), source_scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest())
        if command in {"simulate","probe"}:
            from .test_coverage import development_pending
            declared_scene=load_scene(scene_path,experiment=experiment)
            selected=[test for test in declared_scene["tests"] if test_id is None or test["id"]==test_id]
            if not selected:raise ValueError(f"Unknown test: {test_id}")
            if test_id is None and len(selected)>1:
                result.update(status="TEST_SELECTION_REQUIRED",available_tests=[test["id"] for test in selected],
                              issues=["Choose one complete development test with --test"])
                return result
            pending=development_pending(selected_project,declared_scene["tests"],selected[0])
            if pending and command=="simulate":
                result.update(status="FIRST_PASS_PENDING",available_tests=pending,required_action="run_unvisited_test",
                    issues=["Run each remaining declared test once before retrying this test: "+", ".join(pending)])
                return result
        publish_attempt(workspace,attempt,result,project=selected_project,destination=destination,stage="source_ready")
        geometry_options = {"construction_limits": limits} if limits is not None else {}
        geometry = prepare_geometry(scene_path, attempt / "geometry", **geometry_options)
        if geometry["errors"]:
            resource_limit = any(str(x.get("code", "")).startswith("CONSTRUCTION_") for x in geometry["errors"])
            result.update(status="NEEDS_SIMPLIFICATION" if resource_limit else "DEVELOPMENT_FAILURE", issues=geometry["errors"])
            return result
        result["geometry"] = str(attempt / "geometry" / "geometry.json")
        result["source_bindings"] = geometry.get("source_bindings", [])
        publish_attempt(workspace,attempt,result,project=selected_project,destination=destination,stage="geometry_ready")
        if construction_budget:
            construction_budget.check_available()
        if command == "build":
            destination.mkdir(parents=True, exist_ok=True)
            for item in snapshot.rglob("*"):
                if item.is_file():
                    target = destination / item.relative_to(snapshot)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Replace rather than overwrite: a Harness-owned copy is writable but its timestamps are not.
                    target.unlink(missing_ok=True)
                    shutil.copy2(item, target)
            result.update(status="BUILT", scene=str(destination / "scene.json"))
            return result
        if command == "preview":
            image = render_views(geometry, attempt / "preview.png")
            result.update(status="PREVIEW_READY", image=str(image), model=geometry["model_file"])
            return result
        from .modes import settings as mode_settings
        if command=="check" and selected_project and mode_settings(selected_project):
            from .contracts import load_model_scene
            scene=load_model_scene(scene_path);scene.setdefault("tests",[])
        else:scene = load_scene(scene_path, experiment=experiment)
        prepared = compile_scene(scene, geometry)
        prepared.pop("registered_measurements", None)
        registration = load_registered_measurements(selected_project, scene_path) if selected_project else {}
        if registration: prepared["registered_measurements"] = registration
        write_json_atomic(attempt / "prepared-scene.json", prepared)
        initial = {"time_s": 0., "body_poses": {b["id"]: {
            "position_abs_m": b["position_m"], "rotation_abs_quat_wxyz": b["rotation_wxyz"]} for b in geometry["bodies"]}}
        audit = audit_motion({"bodies":geometry["bodies"]}, [initial], attempt / "initial-audit.json")
        result["initial_audit"] = str(attempt / "initial-audit.json")
        if not audit["passed"]:
            result.update(status="DEVELOPMENT_FAILURE", issues=audit["violations"])
            return result
        from .floating_parts import advisory_report
        assembly = advisory_report(geometry["bodies"], prepared.get("joints") or [], attempt / "floating-parts.json")
        if assembly and assembly["findings"]:
            result["assembly_warnings"] = [finding["message"] for finding in assembly["findings"]]
        if command == "check":
            result.update(status="CHECKED", tests=[t["id"] for t in prepared["tests"]],
                          source_support_checks=prepared.get("source_support_checks", []))
            return result
        tests = [t for t in prepared["tests"] if test_id is None or t["id"] == test_id]
        if not tests: raise ValueError(f"Unknown test: {test_id}")
        if test_id is None and len(tests)>1:
            result.update(status="TEST_SELECTION_REQUIRED",available_tests=[t["id"] for t in tests],
                          issues=["Choose one complete development test with --test. Formal submission runs all approved tests."])
            return result
        if command=="probe":
            from .cost_probe import measure_cost
            from .physics_invocation import _rss_limit
            remaining=construction_budget.check_available()["work_remaining_seconds"]
            with simulation_activity(selected_project):
                result["cost_probe"]=measure_cost(prepared,tests[0],attempt/"cost-probe",python=chrono_python,workspace=workspace,
                    remaining_seconds=remaining,simulation_timeout=simulation_time_steps(limits["simulation_timeout_seconds"])[-1],
                    max_rss_bytes=_rss_limit(selected_project,None))
            result.update(status="COST_PROBE_COMPLETE",required_action=result["cost_probe"]["required_action"])
            return result
        from .test_coverage import development_pending,record_visit,unvisited
        declared={test["id"]:test for test in scene["tests"]}
        summaries = []
        for test in tests:
            active_definition=declared[test["id"]]
            folder = attempt / "tests" / test["id"]; folder.mkdir(parents=True)
            trace_path = folder / "physical.trace.json"
            request = folder / "request.json"
            if construction_budget:
                write_json_atomic(request, {"scene": prepared, "test": test, "output": str(trace_path), "history_binding":model_binding(geometry)})
                command_line = [str(chrono_python), "-X", "utf8", "-m",
                    "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime", str(request)]
                available = construction_budget.check_available()["work_remaining_seconds"]
                execution = run_simulation_bounded(command_line, cwd=workspace, scene=prepared,
                    geometry=geometry, test=test, policy=limits, remaining_seconds=available,
                    report_path=folder / "construction-cost.json", history_dir=selected_project / "construction-simulations")
                if selected_project:record_visit(selected_project,active_definition,"completed" if execution.returncode==0 else "HARNESS_ERROR",folder,True)
                (folder / "stdout.log").write_bytes(execution.stdout)
                (folder / "stderr.log").write_bytes(execution.stderr)
                if execution.returncode:
                    result.update(status="DEVELOPMENT_FAILURE",
                                  issues=[execution.stderr.decode("utf-8", errors="replace")])
                    return result
            else:
                from .budget import physics_budget_session
                from .physics_invocation import run_physics_attempts
                with physics_budget_session(selected_project,workspace=workspace) as budget:
                    timeout=budget.tier.simulation_timeout_seconds if budget.tier else 600
                    invocation=run_physics_attempts(workspace=workspace,project=selected_project,folder=folder,
                        scene=prepared,test=test,chrono_python=chrono_python,budget=budget,
                        timeout_seconds=timeout,runner=run_bounded_subprocess,history_binding=model_binding(geometry),
                        max_attempts=1 if selected_project and len(unvisited(selected_project,list(declared.values())))>1 else 4)
                    if selected_project and (invocation["status"]=="ready" or invocation.get("execution_started")):
                        record_visit(selected_project,active_definition,invocation["status"],folder,True)
                    if invocation["status"]!="ready":
                        result.update(invocation)
                        if invocation["status"]=="ACTIVE_BUDGET_EXHAUSTED" and getattr(budget,"tracker",None):
                            budget.incomplete(issues=invocation.get("issues",[]))
                        return result
                folder=Path(invocation["folder"])
                trace_path=Path(invocation["trace_file"])
            trace = open_trace(trace_path)
            try:
                validate_trace_measurements(trace, registration.get(test["id"]))
                from .trial_cache import store_trial
                cache_record=store_trial(prepared,test,trace_path,chrono_python)
                motion_audit={"passed":False,"status":"deferred_to_formal_validation","reason":"Development returns physical feedback; use audit explicitly or submit for full validation"}
                summaries.append({"test_id": test["id"], "physical_pass": trace["physical_pass"],
                    "numerical_failure_unresolved": trace.get("numerical_failure_unresolved", False),
                    "failure": trace.get("failure"), "assertions": trace.get("assertions", []),
                    "trace_file": str(trace_path), "geometry_audit": motion_audit,"physics_cache":cache_record})
            finally:
                release_trace_cache(trace)
        result.update(tests=summaries, status="DEVELOPMENT_PHYSICS_PASSED" if all(
            t["physical_pass"] for t in summaries) else "DEVELOPMENT_FAILURE")
        if any(t["numerical_failure_unresolved"] for t in summaries): result["status"] = "HARNESS_ERROR"
        return result
    except BudgetExhausted as exc:
        result.update(status="BUDGET_EXHAUSTED", issues=[exc.details], required_action="write_completion_summary")
        return result
    except ConstructionLimitError as exc:
        if selected_project and active_definition and exc.code in {"SIMULATION_HARD_TIMEOUT","SIMULATION_UNCHANGED_TIMEOUT"}:
            from .test_coverage import record_visit
            record_visit(selected_project,active_definition,exc.code,attempt,True)
        result.update(status="BUDGET_EXHAUSTED" if exc.code == "PROJECT_BUDGET_EXHAUSTED" else "NEEDS_SIMPLIFICATION",
                      issues=[exc.details])
        return result
    except ActiveBudgetExhausted as exc:
        result.update(status="ACTIVE_BUDGET_EXHAUSTED",issues=[str(exc)])
        return result
    except BoundedSubprocessError as exc:
        result.update(status="SIMULATION_TIMEOUT" if exc.code=="BOUNDED_SUBPROCESS_TIMEOUT" else "RESOURCE_POLICY_REQUIRED" if exc.code in {"BOUNDED_SUBPROCESS_RSS","BOUNDED_SUBPROCESS_MEMORY_UNAVAILABLE"} else "HARNESS_ERROR",
                      issues=[str(exc)],partial_evidence=str(attempt),memory_budget=getattr(exc,"memory_budget",None),
                      recovery_scope="Preserved files only; a later call starts a new solver run")
        return result
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        result.update(status="HARNESS_ERROR" if isinstance(exc, RuntimeError) else "DEVELOPMENT_FAILURE",
                      issues=[str(exc)])
        return result
    finally:
        activities.close()
        result["elapsed_seconds"] = time.monotonic() - started
        if construction_budget:
            result["construction_budget"] = construction_budget.snapshot()
            from .construction import review_notes
            notes = review_notes(selected_project)
            if notes:
                result["construction_review_notes"] = notes
        write_json_atomic(attempt / "result.json", result)
        publish_attempt(workspace,attempt,result,project=selected_project,
                        destination=locals().get("destination"),stage="finished")
        if construction_budget and selected_project:
            write_json_atomic(selected_project / "construction-evidence" / f"{attempt.name}.json", {
                "schema_version": "construction_evidence_v1", "project": str(selected_project),
                "result_file": str(attempt / "result.json"), "source": str(source),
                "source_scene_sha256": result.get("source_scene_sha256"),
                "scene_snapshot": result.get("scene_snapshot"), "command": command,
                "status": result.get("status"), "elapsed_seconds": result["elapsed_seconds"],
                "test_ids": [test.get("test_id", test.get("id")) if isinstance(test, dict) else test
                             for test in result.get("tests", [])],
            })
