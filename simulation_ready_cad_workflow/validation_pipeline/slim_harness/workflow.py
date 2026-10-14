"""Small unattended author/review/physics loop with program-owned outcomes."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any
import numpy as np

from ..cold_start.runtime_status import RuntimeHeartbeat
from ..cold_start.worker_factory import WORKER_PROVIDERS, create_worker, default_worker_model, normalize_worker_provider
from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import run_bounded_subprocess, BoundedSubprocessError
from .audit import audit_motion, audit_saved_motion, AUDIT_IMPLEMENTATION_SHA256
from .contracts import DeliveryError, load_scene, compile_scene
from .geometry import prepare_geometry, GEOMETRY_IMPLEMENTATION_SHA256
from .implementation import native_implementation_binding
from .numerical import numerical_policy_matches, validate_max_step_seconds, validate_solver_iterations
from .project_isolation import CODE_ROOT
from .evidence import render_views, export_replay
from .print_delivery import load_registered_static_evidence
from .trace_storage import file_digest, open_trace, release_trace_cache, reuse_trace, trace_artifacts
from .measurement_registry import load_registered_measurements, validate_trace_measurements, MeasurementCapabilityError
from .validation_scope import validate_result_scope, validate_validation_scope, ValidationScopeError
from ..trajectory.configuration import model_binding
from .frozen_experiment import load_experiment, author_instructions
from .budget import ActiveBudgetExhausted
from .construction_limits import ConstructionLimitError
from .publication import CODE_ACCEPTANCE_POLICY, LEGACY_ACCEPTANCE_POLICY, code_gate_failures

CHRONO_PYTHON = os.environ.get(
    "SIMREADY_PYCHRONO_EXECUTABLE",
    str(Path.home() / ".conda/envs/chrono_py312_projectchrono/python.exe"),
)


class ActiveWorkflowError(RuntimeError):
    pass


class ProviderRecoveryRequired(RuntimeError):
    def __init__(self, result):
        self.result=result
        super().__init__("REVIEW_PROVIDER_ERROR: "+str(result.get("summary", "Provider recovery required")))


def assert_inactive(project: Path):
    path=Path(project)/"workflow_runtime.json"
    if not path.is_file(): return
    runtime=read(path)
    if runtime.get("status")!="running": return
    try:
        import psutil
        process=psutil.Process(int(runtime["pid"]))
        if process.is_running() and abs(process.create_time()-float(runtime["process_started_at_epoch"]))<.1:
            raise ActiveWorkflowError("This project already has an active runner; its current execution is preserved")
    except (KeyError,TypeError,ValueError):
        return
    except psutil.NoSuchProcess:
        return


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return file_digest(Path(path))


PREVIEW_PATH_FIELDS=("source_path","model_path","scene_path","replay_path","motion_model_path","preview_path","motion_trace")


def recorded_path(workspace,path):
    if not path:
        return path
    path=Path(path).resolve()
    workspace=Path(workspace).resolve()
    return path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else str(path)


def stamp(path,workspace=None):
    path=Path(path).resolve()
    s=path.stat()
    return [recorded_path(workspace,path) if workspace is not None else str(path),s.st_size,s.st_mtime_ns,s.st_ino]


def relativize_preview_paths(workspace,value):
    for item in [value,*(value.get("motions") or [])]:
        for field in PREVIEW_PATH_FIELDS:
            if item.get(field):
                item[field]=recorded_path(workspace,item[field])
    return value



def reuse_initial_geometry_audit(geometry, cached_attempt):
    """Reuse only the unchanged revision's original start pose, never a motion audit.

    The caller has already verified this geometry's scene, source STEP and
    body mesh bindings. Component meshes are additional audit inputs.
    """
    started = time.monotonic()
    if cached_attempt is None:
        return None
    path = cached_attempt/"initial-geometry-audit.json"
    if not path.is_file():
        return None
    previous = read(path)
    if (previous.get("schema_version") != "slim_geometry_audit_v1"
            or previous.get("audit_implementation_sha256") != AUDIT_IMPLEMENTATION_SHA256
            or previous.get("passed") is not True or previous.get("violations") != []
            or previous.get("checked_solver_samples") != 1 or previous.get("available_solver_samples") != 1
            or "validation_scope" in previous):
        return None
    original_geometry = read(cached_attempt/"geometry"/"geometry.json")
    if geometry["bodies"] != original_geometry["bodies"]:
        return None
    for body in geometry["bodies"]:
        for component in body.get("source_components", []):
            path_component = Path(component["mesh_file"])
            if not path_component.is_file() or digest(path_component) != component.get("sha256"):
                return None
    return {**previous, "reused_from": str(path),
            "original_execution_elapsed_seconds": previous.get("original_execution_elapsed_seconds", previous["elapsed_seconds"]),
            "elapsed_seconds": time.monotonic()-started}

class SlimWorkflow:
    def __init__(self,workspace,project,*,executable="codex",model=None,reasoning="medium",chrono_python=CHRONO_PYTHON,physics_timeout_seconds=600,physics_max_step_seconds=.001,physics_solver_iterations=160,tier=None,budget_tracker=None,worker_provider="codex",feedback_mode="full"):
        self._initialize(workspace,project,executable=executable,model=model,reasoning=reasoning,
            chrono_python=chrono_python,physics_timeout_seconds=physics_timeout_seconds,
            physics_max_step_seconds=physics_max_step_seconds,physics_solver_iterations=physics_solver_iterations,
            tier=tier,budget_tracker=budget_tracker,worker_provider=worker_provider,feedback_mode=feedback_mode)

    def _initialize(self,workspace,project,*,executable="codex",model=None,reasoning="medium",chrono_python=CHRONO_PYTHON,physics_timeout_seconds=600,physics_max_step_seconds=.001,physics_solver_iterations=160,tier=None,budget_tracker=None,worker_provider="codex",feedback_mode="full",checkpoint_only=False):
        self._checkpoint_only=checkpoint_only
        self.feedback_mode=feedback_mode or "full"
        self.physics_timeout_seconds=float(physics_timeout_seconds)
        if not math.isfinite(self.physics_timeout_seconds) or self.physics_timeout_seconds<=0:
            raise ValueError("Physics timeout must be positive and finite")
        self.physics_max_step_seconds=validate_max_step_seconds(physics_max_step_seconds)
        self.physics_solver_iterations=validate_solver_iterations(physics_solver_iterations)
        self.workspace=Path(workspace).resolve()
        self.project=Path(project).resolve()
        if not self.project.is_relative_to(self.workspace/".runtime_runs"/"projects"):
            raise ValueError("Slim project must be in the canonical projects directory")
        assert_inactive(self.project)
        from .budget import ProjectBudget, enforce_simulation_timeout
        self.budget=ProjectBudget(self.project,tier,budget_tracker)
        self.tier=self.budget.tier
        self.budget_tracker=self.budget.tracker
        if self.tier is not None:
            self.physics_timeout_seconds=enforce_simulation_timeout(self.tier,self.physics_timeout_seconds)
        self.state=read(self.project/"workflow.json")
        from .construction import recover_author_session, budget_for, worker_for,author_turn_timeout
        if not checkpoint_only:
            self.state=recover_author_session(self.project,self.state)
        settings=self.state.get("request",{})
        worker_provider=normalize_worker_provider(settings.get("worker_provider",worker_provider))
        model=settings.get("model") or model or default_worker_model(worker_provider)
        reasoning=settings.get("reasoning_effort",reasoning)
        chrono_python=settings.get("chrono_python") or chrono_python
        self.construction=budget_for(self.project)
        self.building=bool(settings.get("construction")) and self.construction is not None
        self.budget.state=self.state
        if self.state["schema_version"]!="slim_harness_workflow_v1": raise ValueError("Not a slim workflow")
        self.criteria=read(self.project/"criteria.json")
        experiment=load_experiment(self.project)
        if experiment is not None and self.criteria["criteria"]!=experiment["criteria"]:
            raise ValueError("Criteria differ from the frozen experiment")
        if self.criteria["task_text"]!=(self.project/"task.md").read_text(encoding="utf-8"):
            raise ValueError("Frozen task changed; do not silently reinterpret its criteria")
        lock_path=self.project/"criteria.lock.json"
        self.criteria_sha256=digest(self.project/"criteria.json")
        if lock_path.exists():
            if read(lock_path)["criteria_sha256"]!=self.criteria_sha256:
                raise ValueError("Frozen criteria changed; record a sourced correction before resuming")
        else:
            original=next((event["result"] for event in self.state.get("events",[]) if event.get("kind")=="criteria" and event.get("result",{}).get("status")=="ready"),None)
            if not original or original.get("criteria")!=self.criteria["criteria"] or original.get("summary")!=self.criteria["reviewer_summary"]:
                raise ValueError("Criteria do not match the original independent response")
            write_json_atomic(lock_path,{"schema_version":"slim_criteria_lock_v1","criteria_sha256":self.criteria_sha256,
                                         "task_sha256":digest(self.project/"task.md")})
        from .modes import settings as mode_settings
        self.template_mode=mode_settings(self.project)
        reviewer_reasoning=settings.get("reviewer_reasoning_effort","medium" if self.building or self.template_mode else reasoning)
        self.author=None if checkpoint_only else create_worker(provider=worker_provider,model=model,reasoning_effort=reasoning,executable=executable,
                                  session_id=self.state.get("author_session_id"),disabled_features=("multi_agent","multi_agent_v2"))
        reviewer_provider=normalize_worker_provider(settings.get("reviewer_provider") or worker_provider)
        reviewer_model=settings.get("reviewer_model") or (model if reviewer_provider==worker_provider else default_worker_model(reviewer_provider))
        reviewer_executable=settings.get("reviewer_executable") or (executable if reviewer_provider==worker_provider else "codex")
        self.reviewer=create_worker(provider=reviewer_provider,model=reviewer_model,reasoning_effort=reviewer_reasoning,executable=reviewer_executable)
        from .sandbox_binding import bind_worker
        if self.author is not None:
            self.author=bind_worker(self.author,self.workspace,self.project,'author',settings.get('author_visual_policy','full'))
        self.reviewer=bind_worker(self.reviewer,self.workspace,self.project,'reviewer')
        if self.author is not None:
            self.author=self.budget.worker(self.author)
        self.reviewer=self.budget.worker(self.reviewer)
        if self.author is not None:
            worker_for(self.project,self.author)
        worker_for(self.project,self.reviewer)
        self.chrono_python=chrono_python
        self.modeling_policy_file=self.project/"modeling_policy.json"
        self.modeling_policy=read(self.modeling_policy_file) if self.modeling_policy_file.is_file() else None
        self.modeling_policy_sha256=digest(self.modeling_policy_file) if self.modeling_policy is not None else None
        if self.modeling_policy is not None and self.modeling_policy.get("selected_mode")!="equivalent_elastic":
            raise ValueError("Unsupported selected modeling policy")
        self.modeling_context=({"selected_mode":"equivalent_elastic", "equivalent_elasticity_implemented":True,
            "reference_path":str(CODE_ROOT/"simulation_ready_cad_workflow/validation_pipeline/slim_harness/README.md")}
            if self.modeling_policy is not None else None)
        if self.modeling_context is not None:
            self.modeling_context.update({key:self.modeling_policy[key] for key in (
                "acceptance", "max_test_duration_s", "max_total_test_duration_s", "test_duration_basis")
                if key in self.modeling_policy})
        os.environ["SIMREADY_STEP_SHAPE_CACHE_DIR"]=str(self.workspace/".runtime_runs"/"slim_step_cache")

    def _numerical_cache_matches(self,traces,tests):
        by_id={test["id"]:test for test in tests}
        return bool(traces) and all(row.get("test_id") in by_id
            and numerical_policy_matches(row,by_id[row["test_id"]],self.physics_max_step_seconds,solver_iterations=self.physics_solver_iterations) for row in traces)

    def publish(self,phase,**values):
        self.state.update(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**values)
        self.state["elapsed_seconds"]=(datetime.now(timezone.utc)-datetime.fromisoformat(self.state["started_at"])).total_seconds()
        write_json_atomic(self.project/"workflow.json",self.state)

    def turn(self,kind,call):
        prior=list((self.project/"worker_turns").glob("t*_*"))
        index=max([int(p.name.split("_")[0][1:]) for p in prior]+[0])+1
        directory=self.project/"worker_turns"/f"t{index:03d}_{kind}"
        self.publish(kind,active_turn={"id":directory.name,"directory":str(directory)},stop_reason="")
        result=call(directory)
        result["turn_directory"]=str(directory)
        self.state.setdefault("events",[]).append({"kind":kind,"at":datetime.now(timezone.utc).isoformat(),"result":result})
        if kind=="author" and result.get("status")=="ready":self.state.pop("first_pass_handoff",None)
        self.publish(kind,active_turn={})
        if kind=="author" and self.building and result.get("status")!="ready":
            from .construction import author_turn_timeout
            author_turn_timeout(self.project)
        return result

    def _preview(self,run,geometry,evidence,motions=None):
        out=run/"slim_evidence";out.mkdir(exist_ok=True)
        inputs=[geometry["scene_file"],*[b["source_file"] for b in geometry["bodies"]]]
        motions=[relativize_preview_paths(self.workspace,dict(item)) for item in (motions or [])]
        value={"schema_version":"slim_studio_preview_v1","status":"ready","source_kind":"original_geometry",
               "source_revision":run.name,"model_path":recorded_path(self.workspace,geometry["model_file"]),
               "source_path":recorded_path(self.workspace,geometry["scene_file"]),
               "preview_path":recorded_path(self.workspace,evidence[0]) if evidence else None,
               "pending_update":False,"error":None,
               "part_count":len(geometry["bodies"]),"triangle_count":sum(b.get("triangle_count",0) for b in geometry["bodies"]),
               "input_stamps":[stamp(p,self.workspace) for p in dict.fromkeys(inputs)],
               "input_version":digest(geometry["scene_file"]),"motions":motions,
               "validation_scope":geometry.get("validation_scope")}
        if motions:
            from .motion_preview import best_motion
            value.update({k:v for k,v in (best_motion(motions) or motions[0]).items() if k not in {"id","label","load_case_id"}})
        write_json_atomic(out/"preview.json",value)


    def _motion(self,geometry,trace,trace_path):
        assets=export_replay(geometry,trace,trace_path.parent/"replay")
        identifier=trace["test_id"]
        scope=trace.get("validation_scope")
        motion={"id":identifier,"load_case_id":identifier,"label":identifier.replace("_"," "),
                "motion_model_path":assets["model_file"],"scene_path":assets["scene_file"],"replay_path":assets["replay_file"],
                "motion_trace":str(trace_path),"motion_frame_count":min(720,len(trace["samples"])),"replay_version":digest(trace_path)}
        if scope:
            motion["validation_scope"]=scope
            motion["label"] += " / " + scope["display_name"]
        return assets,motion



    def _finish_validation(self,run,scene,geometry,evidence,summaries,source_bindings,attempt,started,*,resumed_from=None):
        """Current policy: deterministic gates own the result; no model verdict."""
        scope=geometry.get("validation_scope")
        validate_result_scope({"validation_scope":scope,"tests":summaries},scene)
        failures=code_gate_failures(scene,summaries)
        _,static_errors=load_registered_static_evidence(self.project,run/"design"/"scene.json")
        if failures or static_errors:
            return {"status":"NEEDS_REVISION","issues":[*failures,*static_errors],"attempt":str(attempt)}
        if digest(self.project/"criteria.json")!=self.criteria_sha256 or any(digest(path)!=value for path,value in source_bindings.items()):
            return {"status":"HARNESS_ERROR","issues":["Source changed during validation; repeat against one immutable revision"],"attempt":str(attempt)}
        if self.state.get("first_pass_handoff"):
            return {"status":"NEEDS_REVISION","issues":["First test coverage completed from an interrupted author delivery; author completion is still required"],
                    "attempt":str(attempt),"tests":summaries,"completed_tests":summaries,"pending_tests":[],
                    "first_pass_only":True,"complete_instance_pass":False}
        model_scope=scope["display_name"]+"; "+scope["dynamic_geometry"] if scope else "dry rigid-body mechanical task"
        model_scope+="; code validation of declared tests and registered measurements; original geometry audited at uniform 10 Hz saved poses with exact endpoints; no continuous-time clearance claim or final model review"
        if scene.get("spur_gears"):
            from .spur_gears import SPUR_MODEL_SCOPE
            model_scope+="; "+SPUR_MODEL_SCOPE
        return {"status":"PASS","issues":[],"attempt":str(attempt),"tests":summaries,
                "acceptance_policy":CODE_ACCEPTANCE_POLICY,
                "source_bindings":source_bindings,"criteria_sha256":self.criteria_sha256,
                "revision_execution_seconds":time.monotonic()-started,"resumed_physical_evidence_from":resumed_from,
                "validation_scope":scope,"scope":model_scope}

    def run_revision(self):
        if self.template_mode:
            from .template_workflow import run_revision
            return run_revision(self)
        if self.building:self.construction.check_available()
        if digest(self.project/"criteria.json")!=self.criteria_sha256:
            raise RuntimeError("Frozen criteria changed during the workflow")
        run=self.project/"runs"/f"r{int(self.state['revision']):02d}"
        scene_path=run/"design"/"scene.json"
        number=len(list((run/"attempts").glob("a[0-9]*")))+1
        attempt=run/"attempts"/f"a{number:03d}"
        attempt.mkdir(parents=True)
        write_json_atomic(attempt/"acceptance-policy.json",{
            "policy":CODE_ACCEPTANCE_POLICY,"prior_policy":self.state.get("acceptance_policy",LEGACY_ACCEPTANCE_POLICY),
            "authorization":"User requested model review only for changes to simulation tasks; remove initial, delivery-preflight and final model reviews.",
            "hard_gates_preserved":True,"historical_results_reinterpreted":False})
        started=time.monotonic()
        self.publish("geometry",stop_reason="",acceptance_policy=CODE_ACCEPTANCE_POLICY)
        experiment=load_experiment(self.project)
        scene=load_scene(scene_path,experiment=experiment)
        if self.building:
            from .construction_limits import scene_preflight,check_task_test_counts
            config=self.construction.snapshot()["config"]
            check_task_test_counts(scene,config,
                task_count=sum(c.get("kind")=="function" for c in self.criteria["criteria"]))
            scene_preflight(scene,config)
            plan=read(run/"design"/"construction_plan.json") if (run/"design"/"construction_plan.json").exists() else {}
            write_json_atomic(attempt/"construction-plan.json",{"budget":self.construction.snapshot(),
                "proposed_scope":plan,"tests":scene["tests"],
                "core_criteria":self.criteria["criteria"]})
        if experiment is not None and read(scene_path)!=scene:
            write_json_atomic(scene_path,scene)
        baseline=self.project/"simulation-task-baseline.json"
        if not baseline.is_file():
            from .task_adaptation import task_definition
            write_json_atomic(baseline,task_definition((self.project/"task.md").read_text(encoding="utf-8"),scene))
        if self.tier is not None:
            from .physical_parts import is_time_only_continuation
            if not is_time_only_continuation(self.project, self.tier):
                from .budget import check_part_count, check_task_count
                part_violations=check_part_count(scene.get("bodies",[]),self.tier)
                if part_violations:
                    return {"status":"NEEDS_REVISION","issues":part_violations,"attempt":str(attempt),"budget_violation":"part_count"}
                task_violations=check_task_count(scene.get("tests",[]),self.tier)
                if task_violations:
                    return {"status":"NEEDS_REVISION","issues":task_violations,"attempt":str(attempt),"budget_violation":"task_count"}
        if self.budget_tracker is not None and self.budget_tracker.is_exhausted:
            from .budget import build_incomplete_summary
            return build_incomplete_summary(tier=self.tier,tracker=self.budget_tracker,
                completed_tests=[],pending_tests=[t["id"] for t in scene.get("tests",[])],
                issues=["Active budget exhausted before geometry preparation"],evidence_paths=[],blockers=[])
        if self.modeling_policy is not None:
            feedback=self.modeling_policy.get("source_review_feedback") or {}
            bindings=feedback.get("source_bindings",{})
            if feedback.get("verdict")=="REVISE" and feedback.get("findings") and bindings:
                sources=[((scene_path.parent/relative).resolve(),expected) for relative,expected in bindings.items()]
                if all(path.is_relative_to(scene_path.parent) and path.is_file() and digest(path)==expected
                       for path,expected in sources):
                    raise DeliveryError("Independent source review requires a new author revision: "+json.dumps(feedback,ensure_ascii=False))
            per_test_limit=self.modeling_policy.get("max_test_duration_s")
            total_limit=self.modeling_policy.get("max_total_test_duration_s")
            overlong=[{"id":test["id"],"duration_s":test["duration_s"]} for test in scene["tests"]
                      if per_test_limit is not None and test["duration_s"]>per_test_limit]
            total_duration=sum(test["duration_s"] for test in scene["tests"])
            if overlong or (total_limit is not None and total_duration>total_limit):
                raise DeliveryError("Current user-authorized compact-test planning policy requires a new author revision. "
                    +json.dumps({"max_test_duration_s":per_test_limit,"max_total_test_duration_s":total_limit,
                                 "actual_total_test_duration_s":total_duration,"overlong_tests":overlong},ensure_ascii=False)
                    +" Preserve source geometry/detail and all required functions, with real controls, triggers, contact/impact and continuous-running evidence. "
                    "Replan operations and observation windows; do not merely truncate durations, split the original long tests into short segments, or enlarge solver steps. "
                    "Keep the original long tests for separate stability revalidation. Changed measurement windows require time-limited task-change advice and an author decision; existing registered measurements still require actual re-registration before execution. "
                    "These are current planning limits; physical event evidence may justify a supervising policy revision. "
                    +"Acceptance: "+str(self.modeling_policy.get("acceptance","Preserve all required task functions."))
                    +" Basis: "+str(self.modeling_policy.get("test_duration_basis","User-authorized compact functional testing with simulation speed considered.")))
        from .static_source_metadata import grouped_manifest_findings
        source_findings=grouped_manifest_findings(scene_path,self.criteria)
        if source_findings:
            source_report=attempt/"source-delivery-metadata.json"
            write_json_atomic(source_report,{"findings":source_findings,"scope":"Source metadata only; no F7 execution receipt or physical outcome."})
            raise DeliveryError("Source delivery metadata requires revision: "+json.dumps(source_findings,ensure_ascii=False)+" Evidence: "+str(source_report))
        inventory_tier = self.tier
        if inventory_tier is not None:
            from .physical_parts import (
                PART_INVENTORY_REQUIRED,
                check_physical_part_inventory,
                resolve_physical_part_inventory,
            )
            inventory=resolve_physical_part_inventory(
                scene_path, scene,
                task_text=(self.project/"task.md").read_text(encoding="utf-8"),
                project=self.project)
            write_json_atomic(attempt/"physical-part-inventory.json", inventory)
            if inventory.get("status")==PART_INVENTORY_REQUIRED:
                return {"status":"NEEDS_REVISION","issues":inventory.get("issues",[PART_INVENTORY_REQUIRED]),
                        "attempt":str(attempt),"budget_violation":"physical_part_inventory"}
            inventory_violations=check_physical_part_inventory(inventory,inventory_tier,project=self.project)
            if inventory_violations:
                return {"status":"NEEDS_REVISION","issues":inventory_violations,
                        "attempt":str(attempt),"budget_violation":"physical_part_inventory"}
        # Establish/compare the task baseline before geometric preparation, but
        # do not spend model time on exhausted budgets or invalid inventories.
        advice=self._task_change_advice(scene_path,scene)
        if advice is not None:return advice
        from .step_delivery import refresh_registered_step_delivery
        delivery=refresh_registered_step_delivery(self.project,scene_path,attempt)
        if delivery is not None and delivery.get("status")!="ready":
            return {"status":"NEEDS_REVISION","issues":delivery.get("issues",[]),"attempt":str(attempt)}
        current_policy_binding=({"source_file":str(self.modeling_policy_file),"sha256":self.modeling_policy_sha256,"selected_mode":self.modeling_policy["selected_mode"]}
                                if self.modeling_policy is not None else None)
        measurement_bindings=load_registered_measurements(self.project,scene_path)
        native_implementation=native_implementation_binding()
        cached_attempt=None
        cached_geometry=None
        for candidate in sorted((run/"attempts").glob("a*/geometry/geometry.json"),reverse=True):
            value=read(candidate)
            if value.get("scene_sha256")!=digest(scene_path) or value.get("errors"): continue
            if value.get("geometry_implementation_sha256")!=GEOMETRY_IMPLEMENTATION_SHA256: continue
            if not value.get("bodies") or any(not {"source_file","source_sha256","mesh_file","mesh_sha256"}<=set(b) for b in value["bodies"]): continue
            if any(not Path(b["source_file"]).is_file() or digest(b["source_file"])!=b["source_sha256"]
                   or not Path(b["mesh_file"]).is_file() or digest(b["mesh_file"])!=b["mesh_sha256"] for b in value["bodies"]): continue
            cached_attempt=candidate.parent.parent;cached_geometry=value
            break
        # Failed traces are reused per test below. A prior failure cannot
        # short-circuit the rest of this revision's declared test set.
        for bundle_file in sorted((run/"attempts").glob("a*/validated-physics.json"),reverse=True):
            bundle=read(bundle_file)
            if (self.modeling_policy_sha256 is not None and
                    bundle.get("source_bindings",{}).get(str(self.modeling_policy_file))!=self.modeling_policy_sha256): continue
            try: validate_result_scope({"validation_scope":bundle.get("geometry",{}).get("validation_scope"),"tests":bundle.get("tests",[])},scene)
            except ValidationScopeError: continue
            if bundle.get("criteria_sha256")!=self.criteria_sha256: continue
            if bundle.get("geometry",{}).get("geometry_implementation_sha256")!=GEOMETRY_IMPLEMENTATION_SHA256: continue
            if not bundle.get("tests") or any(row.get("execution_implementation")!=native_implementation
                or row.get("geometry_audit",{}).get("audit_implementation_sha256")!=AUDIT_IMPLEMENTATION_SHA256
                for row in bundle["tests"]): continue
            if not self._numerical_cache_matches(bundle["tests"],scene["tests"]): continue
            if any(not Path(p).is_file() or digest(p)!=expected for p,expected in bundle["source_bindings"].items()): continue
            if any(not Path(p).is_file() or file_digest(Path(p),drop_cache=True)!=expected for p,expected in bundle["evidence_bindings"].items()): continue
            try:
                for row in bundle["tests"]:
                    validate_trace_measurements(row,measurement_bindings.get(row.get("test_id")))
            except MeasurementCapabilityError: continue
            geometry=bundle["geometry"]
            write_json_atomic(attempt/"geometry"/"geometry.json",geometry)
            evidence=[Path(p) for p in bundle["images"] if Path(p).is_file()]
            if (run/"additional-evidence.json").exists():
                extra=read(run/"additional-evidence.json")
                if extra.get("scene_sha256")==digest(scene_path): evidence.extend(Path(p) for p in extra.get("images",[]) if Path(p).is_file())
            self._preview(run,geometry,evidence,bundle["motions"])
            return self._finish_validation(run,scene,geometry,evidence,bundle["tests"],bundle["source_bindings"],
                                       attempt,started,resumed_from=str(bundle_file))
        geometry=cached_geometry or prepare_geometry(scene_path,attempt/"geometry",**(
            {"construction_limits":self.construction.snapshot()["config"]} if self.building else {}))
        if self.building:
            from .construction_limits import geometry_preflight
            geometry_preflight(geometry,self.construction.snapshot()["config"])
        if cached_geometry: write_json_atomic(attempt/"geometry"/"geometry.json",geometry)
        if geometry["errors"]:
            return {"status":"NEEDS_REVISION","issues":geometry["errors"],"attempt":str(attempt)}
        source_bindings={str(scene_path):digest(scene_path),**{b["source_file"]:b["source_sha256"] for b in geometry["bodies"]}}
        if experiment is not None:
            source_bindings[str(self.project/"frozen_experiment.json")]=self.state["request"]["frozen_experiment_sha256"]
        if self.modeling_policy_sha256 is not None:
            source_bindings[str(self.modeling_policy_file)]=self.modeling_policy_sha256
        if measurement_bindings:
            registration_path=self.project/"measurement-registration.json"
            source_bindings[str(registration_path)]=digest(registration_path)
            source_bindings[str(self.project/"task.md")]=digest(self.project/"task.md")
            for binding in measurement_bindings.values():
                source_bindings.update({str(scene_path.parent/relative):expected
                    for relative,expected in binding["source_plan_bindings"].items()})
        cached_view=cached_attempt/"overview.png" if cached_attempt else None
        evidence=[cached_view if cached_view and cached_view.is_file() else render_views(geometry,attempt/"overview.png")]
        additional_file=run/"additional-evidence.json"
        if additional_file.exists():
            additional=read(additional_file)
            if additional.get("scene_sha256")==digest(scene_path):
                evidence.extend(Path(p) for p in additional.get("images",[]) if Path(p).is_file())
        self._preview(run,geometry,evidence)
        initial={"time_s":0.,"body_poses":{b["id"]:{"position_abs_m":b["position_m"],"rotation_abs_quat_wxyz":b["rotation_wxyz"]} for b in geometry["bodies"]}}
        initial_audit=reuse_initial_geometry_audit(geometry,cached_attempt) if cached_geometry else None
        if initial_audit is None:
            initial_audit=audit_motion({"bodies":geometry["bodies"]},[initial],attempt/"initial-geometry-audit.json")
        else:
            write_json_atomic(attempt/"initial-geometry-audit.json",initial_audit)
        issues=[]
        try:
            prepared=compile_scene(scene,geometry)
            if self.modeling_policy is not None and not prepared.get("equivalent_elasticity"):
                raise DeliveryError("The selected equivalent-elastic policy requires an author-supplied equivalent_elasticity declaration with source-based parameters and scope. The installed interface is described in the current slim_harness/README.md.")
        except DeliveryError as exc:
            return {"status":"HARNESS_ERROR" if str(exc).startswith("CAPABILITY_GAP:") else "NEEDS_REVISION",
                    "issues":[str(exc)],"attempt":str(attempt)}
        validation_scope=prepared.get("validation_scope") if prepared is not None else None
        if validation_scope is not None:
            geometry["validation_scope"]=validation_scope
            source_bindings.update(validation_scope["source_bindings"])
            write_json_atomic(attempt/"geometry"/"geometry.json",geometry)
        if prepared is not None:
            if self.modeling_policy_sha256 is not None:
                prepared["modeling_policy_binding"]=current_policy_binding
            prepared.pop("registered_measurements",None)
            if measurement_bindings: prepared["registered_measurements"]=measurement_bindings
        if not initial_audit["passed"]: issues.extend(initial_audit["violations"])
        if prepared is not None:
            write_json_atomic(attempt/"prepared-scene.json",prepared)
            if validation_scope is not None: self._preview(run,geometry,evidence)
        if issues:
            return {"status":"NEEDS_REVISION","issues":issues,"attempt":str(attempt)}
        # Structural, initial-source and compilation gates above own readiness.
        # No delivery-preflight model or invented test_coverage is consulted.
        _,static_errors=load_registered_static_evidence(self.project,scene_path)
        if static_errors:
            return {"status":"NEEDS_REVISION","issues":static_errors,"attempt":str(attempt)}
        write_json_atomic(attempt/"prepared-scene.json",prepared)
        from .test_coverage import ordered_tests, record_visit, coverage_summary, planned_tests
        from .budget import ActiveBudgetExhausted
        from .construction_budget import BudgetExhausted
        summaries,motions=[],[]
        declared={test["id"]:test for test in (planned_tests(self.project) or scene["tests"])}
        prepared_tests={test["id"]:test for test in prepared["tests"]}
        tests=[prepared_tests[test["id"]] for test in ordered_tests(self.project,list(declared.values()))]
        remaining_test_ids=[test["id"] for test in tests]
        global_stop=None
        # First give every independent test a physical execution opportunity.
        # Audits, rendering and author repairs cannot hold up later tests.
        for test in tests:
            try:
                self.budget.require_available()
                if self.building:self.construction.check_available()
            except (ActiveBudgetExhausted,BudgetExhausted) as exc:
                global_stop=str(exc)
                break
            self.publish("physics",active_test=test["id"])
            folder=attempt/"tests"/test["id"];folder.mkdir(parents=True)
            try:
                summary=self._execute_physical_test(prepared,geometry,test,folder,cached_attempt,
                    native_implementation,measurement_bindings,max_attempts=4 if len(remaining_test_ids)==1 else 1)
            except (ActiveBudgetExhausted,BudgetExhausted) as exc:
                global_stop=str(exc)
                break
            except ConstructionLimitError as exc:
                summary=self._test_failure(test,"SIMULATION_TIMEOUT" if "TIMEOUT" in exc.code else "HARNESS_ERROR",[exc.details],folder,execution_started=exc.code in {"SIMULATION_HARD_TIMEOUT","SIMULATION_UNCHANGED_TIMEOUT"})
            except BoundedSubprocessError as exc:
                resource=exc.code in {"BOUNDED_SUBPROCESS_RSS","BOUNDED_SUBPROCESS_MEMORY_UNAVAILABLE"}
                summary=self._test_failure(test,"RESOURCE_POLICY_REQUIRED" if resource else "HARNESS_ERROR",[str(exc)],folder,
                    details={"memory_budget":getattr(exc,"memory_budget",None)},execution_started=exc.code=="BOUNDED_SUBPROCESS_RSS")
            except Exception as exc:
                summary=self._test_failure(test,"HARNESS_ERROR",[f"{type(exc).__name__}: {exc}"],folder)
            summaries.append(summary)
            remaining_test_ids.remove(test["id"])
            record_visit(self.project,declared[test["id"]],summary["execution_status"],folder,summary.get("execution_started",False))
            write_json_atomic(folder/"test-outcome.json",summary)
            write_json_atomic(attempt/"test-coverage.json",coverage_summary(tests,summaries,remaining_test_ids))
            if summary.get("failure_status")=="ACTIVE_BUDGET_EXHAUSTED":
                global_stop="Active budget exhausted during a test"
                break
        # Every physical opportunity above is retained even if later auditing
        # or replay creation fails. A failed test never grants acceptance.
        for summary in summaries:
            if not summary.get("trace_file") or summary.get("failure_status")=="HARNESS_ERROR":continue
            trace=None
            try:
                trace_path=Path(summary["trace_file"]);folder=trace_path.parent
                trace=open_trace(trace_path)
                assets,motion=self._motion(geometry,trace,trace_path)
                summary["replay"]=assets;motions.append(motion)
                self._preview(run,geometry,evidence,motions)
                if not summary.get("physical_pass"):
                    summary["geometry_audit"]={"passed":False,"status":"not_completed_for_rejected_test",
                        "reason":"Physical failure already prevents acceptance; all other tests received their first execution opportunity"}
                    write_json_atomic(folder/"test-outcome.json",summary)
                    continue
                self.publish("geometry_audit",active_test=summary["test_id"])
                if self.building:
                    from .construction_audit import run_motion_audit_bounded
                    snap=self.construction.check_available()
                    audit=run_motion_audit_bounded(attempt/"prepared-scene.json",trace_path,folder/"geometry-audit.json",
                        policy=snap["config"],remaining_seconds=snap["work_remaining_seconds"],
                        cwd=self.workspace,test_id=summary["test_id"])
                else:
                    audit=audit_saved_motion(prepared,trace["samples"],folder/"geometry-audit.json")
                summary["geometry_audit"]=audit
                evidence.append(render_views(geometry,folder/"final-pose.png",trace["samples"][-1]["body_poses"]))
                if not audit["passed"]:summary.setdefault("failure_status","NEEDS_REVISION")
            except (ActiveBudgetExhausted,BudgetExhausted) as exc:
                global_stop=str(exc)
                break
            except ConstructionLimitError as exc:
                summary["geometry_audit"]={"passed":False,"status":"not_completed","reason":str(exc),"details":exc.details}
                summary.setdefault("failure_status","NEEDS_REVISION")
            except Exception as exc:
                summary["geometry_audit"]={"passed":False,"status":"audit_error","reason":f"{type(exc).__name__}: {exc}"}
                summary.setdefault("failure_status","HARNESS_ERROR")
            finally:
                if trace is not None:release_trace_cache(trace)
            write_json_atomic(Path(summary["trace_file"]).parent/"test-outcome.json",summary)
        coverage=coverage_summary(tests,summaries,remaining_test_ids)
        write_json_atomic(attempt/"test-coverage.json",coverage)
        failures=[row for row in summaries if not row.get("physical_pass") or not row.get("geometry_audit",{}).get("passed")]
        if global_stop or failures or remaining_test_ids:
            statuses={row.get("failure_status") for row in failures}
            status=("ACTIVE_BUDGET_EXHAUSTED" if global_stop else "HARNESS_ERROR" if "HARNESS_ERROR" in statuses
                    else "RESOURCE_POLICY_REQUIRED" if "RESOURCE_POLICY_REQUIRED" in statuses
                    else "SIMULATION_TIMEOUT" if "SIMULATION_TIMEOUT" in statuses else "NEEDS_REVISION")
            return {"status":status,"issues":[*failures,*([global_stop] if global_stop else [])],
                    "attempt":str(attempt),"tests":summaries,"completed_tests":summaries,
                    "pending_tests":coverage["pending_completion_test_ids"],"unrun_tests":coverage["not_run_test_ids"],"test_coverage":coverage}
        bound_files=[path for row in summaries for path in trace_artifacts(row["trace_file"])]
        bound_files.extend(Path(row["trace_file"]).parent/"geometry-audit.json" for row in summaries)
        write_json_atomic(attempt/"validated-physics.json",{"criteria_sha256":self.criteria_sha256,"geometry":geometry,
                          "source_bindings":source_bindings,"evidence_bindings":{str(p):file_digest(p,drop_cache=True) for p in bound_files},
                          "images":[str(p) for p in evidence],"motions":motions,"tests":summaries,"acceptance_policy":CODE_ACCEPTANCE_POLICY})
        return self._finish_validation(run,scene,geometry,evidence,summaries,source_bindings,attempt,started)

    def _test_failure(self,test,status,issues,folder,details=None,execution_started=False):
        return {"test_id":test["id"],"execution_status":status if execution_started else "not_run",
                "execution_started":execution_started,"physical_pass":False,
                "failure_status":status,"failure":{"issues":issues},"geometry_audit":{"passed":False,"status":"not_run"},
                "evidence_folder":str(folder),"execution_details":details or {}}

    def _execute_physical_test(self,prepared,geometry,test,folder,cached_attempt,native_implementation,measurement_bindings,max_attempts=1):
        trace_path=folder/"physical.trace.json"
        request=folder/"request.json"
        # Construction runs are bounded by the project's ledger, not the tier table,
        # and extend past their configured limit.
        from .construction_limits import simulation_time_steps
        effective_timeout=(simulation_time_steps(self.construction.snapshot()["config"]["simulation_timeout_seconds"])[-1]
                           if self.building else self.budget.limit(self.physics_timeout_seconds))
        execution_limits={"timeout_seconds":effective_timeout,"max_rss_bytes":"auto"}
        numerical_settings={"max_step_s":self.physics_max_step_seconds,"solver_iterations":self.physics_solver_iterations}
        write_json_atomic(request,{"scene":prepared,"test":test,"output":str(trace_path),
                                   "execution_limits":execution_limits,"numerical_settings":numerical_settings,
                                   "history_binding":model_binding(geometry)})
        previous_folder=cached_attempt/"tests"/test["id"] if cached_attempt else None
        trace=None
        if previous_folder and (previous_folder/"request.json").is_file() and (previous_folder/"physical.trace.json").is_file():
            prior_request=read(previous_folder/"request.json")
            prior_trace=open_trace(previous_folder/"physical.trace.json")
            try:
                validate_trace_measurements(prior_trace,measurement_bindings.get(test["id"]))
                measurement_cache_matches=True
            except MeasurementCapabilityError: measurement_cache_matches=False
            if (prior_request.get("scene")==prepared and prior_request.get("test")==test
                and prior_trace.get("test_id")==test["id"]
                and prior_trace.get("execution_implementation")==native_implementation
                and numerical_policy_matches(prior_trace,test,self.physics_max_step_seconds,solver_iterations=self.physics_solver_iterations)
                and measurement_cache_matches
                ):
                trace=reuse_trace(previous_folder/"physical.trace.json",trace_path)
            release_trace_cache(prior_trace)
        if trace is None:
            from .trial_cache import reuse_trial
            trace=reuse_trial(self.workspace,prepared,test,trace_path,self.chrono_python,max_step_s=self.physics_max_step_seconds,solver_iterations=self.physics_solver_iterations)
        if trace is None and self.building:
            from .construction_limits import run_simulation_bounded
            snap=self.construction.check_available()
            command=[self.chrono_python,"-X","utf8","-m","simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",str(request)]
            execution=run_simulation_bounded(command,scene=prepared,geometry=geometry,test=test,
                policy=snap["config"],remaining_seconds=snap["work_remaining_seconds"],cwd=self.workspace,
                report_path=folder/"construction-simulation.json",history_dir=self.project/"construction-simulations")
            (folder/"stdout.log").write_bytes(execution.stdout)
            (folder/"stderr.log").write_bytes(execution.stderr)
            if execution.returncode!=0:
                return self._test_failure(test,"HARNESS_ERROR",[execution.stderr.decode("utf-8",errors="replace")],folder,execution_started=True)
        elif trace is None:
            from .physics_invocation import run_physics_attempts
            invocation=run_physics_attempts(workspace=self.workspace,project=self.project,folder=folder,
                scene=prepared,test=test,chrono_python=self.chrono_python,budget=self.budget,
                timeout_seconds=self.physics_timeout_seconds,numerical_settings=numerical_settings,
                history_binding=model_binding(geometry),
                modeling_policy=self.modeling_policy,runner=run_bounded_subprocess,max_attempts=max_attempts)
            if invocation["status"]!="ready":
                return self._test_failure(test,invocation["status"],invocation.get("issues",[]),folder,invocation,invocation.get("execution_started",False))
            folder=Path(invocation["folder"])
            trace_path=Path(invocation["trace_file"])
        if trace is None:
            try:
                trace=open_trace(trace_path)
            except Exception as exc:
                return self._test_failure(test,"HARNESS_ERROR",[f"{type(exc).__name__}: {exc}"],folder,execution_started=True)
        if trace.get("test_id")!=test["id"]:
            release_trace_cache(trace)
            return self._test_failure(test,"HARNESS_ERROR",["Trace belongs to a different test"],folder,execution_started=True)
        if trace.get("measurement_error"):
            problem=trace["measurement_error"];release_trace_cache(trace)
            return self._test_failure(test,"HARNESS_ERROR",[problem],folder,execution_started=True)
        try: validate_trace_measurements(trace,measurement_bindings.get(test["id"]))
        except MeasurementCapabilityError as exc:
            release_trace_cache(trace)
            return self._test_failure(test,"HARNESS_ERROR",[str(exc)],folder,execution_started=True)
        summary={key:value for key,value in trace.items() if key not in {"samples","signals"}}
        summary.update(test_id=test["id"],trace_file=str(trace_path),execution_status="reused" if trace.get("reused_from") else "completed",execution_started=True,
                       geometry_audit={"passed":False,"status":"pending_audit"})
        if not trace.get("physical_pass"):
            summary["failure_status"]="HARNESS_ERROR" if trace.get("numerical_failure_unresolved") else "NEEDS_REVISION"
        release_trace_cache(trace)
        return summary

    def _annotate_budget(self, result):
        """Attach budget timing to a result when a tracker is active."""
        result.setdefault("acceptance_policy",CODE_ACCEPTANCE_POLICY)
        if self.state.get("task_adjustments"):
            result["task_adjustments"]=self.state["task_adjustments"]
            result["acceptance_target"]="author_decided_adjusted_task"
            result["original_task_pass_claimed"]=False
        if result.get("status")!="PASS" and "test_coverage" not in result:
            from .test_coverage import planned_tests,coverage_summary
            tests=planned_tests(self.project) or []
            rows=result.get("tests",result.get("completed_tests",[]))
            seen={row.get("test_id",row.get("id")) for row in rows}
            pending=[test["id"] for test in tests if test["id"] not in seen]
            result["test_coverage"]=coverage_summary(tests,rows,pending)
            result.setdefault("unrun_tests",result["test_coverage"]["not_run_test_ids"])
            result.setdefault("pending_tests",pending)
        return self.budget.annotate(result)

    def _task_change_advice(self, scene_path, scene):
        from .task_adaptation import task_definition, review_task_change
        definition=task_definition((self.project/"task.md").read_text(encoding="utf-8"),scene)
        baseline=self.project/"simulation-task-baseline.json"
        if not baseline.is_file():
            write_json_atomic(baseline,definition)
            return None
        if read(baseline)==definition:
            return None
        scene_before=digest(scene_path)
        task_before=digest(self.project/"task.md")
        timing=self.budget.annotate({}).get("budget_timing",{})
        feedback=self.state.get("latest_feedback",{})
        decision=self.turn("task_change_review",lambda directory:review_task_change(
            self.reviewer,self.author,self.workspace,self.project,scene_path.parent,directory,definition,
            context={"original_task":self.state.get("request",{}).get("brief"),
                     "remaining_active_seconds":timing.get("remaining_active_seconds"),"feedback":feedback},
            tier_name=self.tier.name if self.tier else "light",budget=self.budget))
        self.state["author_session_id"]=self.author.session_id
        if digest(scene_path)!=scene_before or digest(self.project/"task.md")!=task_before or digest(self.project/"criteria.json")!=self.criteria_sha256:
            return {"status":"NEEDS_REVISION","issues":["Task changed during its decision; submit the new definition"],"complete_instance_pass":False}
        if not decision.get("accepted"):
            return {"status":"NEEDS_REVISION","issues":["Author must revise or retain the simulation task"],
                    "task_change_decision":decision,"complete_instance_pass":False}
        write_json_atomic(baseline,definition)
        return None

    def _budget_revision(self, design, feedback):
        """Let the author choose the tradeoff after bounded task-change advice."""
        from .feedback_ablation import transform_feedback
        feedback=transform_feedback(feedback,self.feedback_mode)
        from .task_adaptation import (decision_context,decision_prompt,feedback_text,record_decision,
                                      revision_response_schema,review_and_apply_adjustment)
        attempt=0
        while True:
            self.budget.require_available()
            timing=self.budget.annotate({}).get("budget_timing",{"remaining_active_seconds":None})
            context=decision_context(self.state,timing,feedback)
            if self.template_mode:
                from .task_adaptation import BUDGET_ACTIONS
                context.update(adjustment_allowed=True,allowed_actions=list(BUDGET_ACTIONS),
                               adjustment_scope="reviewer_filled_template_parameters")
            context_path=design.parent/f"decision-context-{attempt:02d}.json"
            write_json_atomic(context_path,context)
            prompt=(f"Continue the same machine in {design}; prior revisions remain preserved. "
                    +decision_prompt(context,context_path)+author_instructions(self.project)+
                    "For revise_mechanism or adjust_execution, implement your chosen source/model correction within the existing allowed modeling scope. "
                    "For adjust_task, write task_adjustment.json here with task_text, reason and optional source-quoted criteria, "
                    "and update scene.json with the complete corresponding proposed simulation tests. "
                    "Do not directly edit task.md, criteria, locks or results. A time-limited reviewer advises ONLY on this task change; "
                    "you then decide whether to accept it, revise it or retain the old task. "
                    "For finish_partial, retain current work and explain the remaining functional problem and accepted tradeoff. "
                    "Do not change Harness code or physics/geometry safety gates. Return the required budget_decision and normal response fields.\n"
                    f"CURRENT CRITERIA:\n{json.dumps(self.criteria,ensure_ascii=False)}")
            from .construction import author_turn_timeout,policy_prompt
            if self.building:prompt+=policy_prompt(self.project)
            if self.template_mode:
                from .modes import author_instructions as mode_instructions
                prompt=(f"Continue the mechanical design in {design}; retain prior revisions. "+decision_prompt(context,context_path)
                        +mode_instructions(self.project)+"For adjust_task, write template_adjustment.json with parameters and reason; "
                        "the Harness reviewer fills binding.py. Do not edit task.md, criteria or reviewer outputs.\n"
                        +f"CURRENT FEEDBACK: {feedback_text(feedback,context_path)}")
            response=self.turn("author",lambda directory:self.author.run_turn(
                design,prompt,directory,author_turn_timeout(self.project),response_schema=revision_response_schema()))
            self.state["author_session_id"]=self.author.session_id
            if response.get("status")=="provider_error":return response
            try:
                record=record_decision(design.parent/f"budget-decision-{attempt:02d}.json",response,context)
            except ValueError as exc:
                self.publish("author_followup",stop_reason="budget_decision_required",active_turn={})
                return {"status":"EVIDENCE_REQUIRED","issues":[str(exc)],"revision":self.state["revision"]}
            action=record["budget_decision"]["action"]
            if action=="finish_partial":
                result=self._annotate_budget({"schema_version":"slim_harness_result_v1",
                    "status":"INCOMPLETE_EVIDENCE","revision":self.state["revision"],
                    "complete_instance_pass":False,"budget_decision":record,
                    "issues":feedback.get("issues",[]),"completed_tests":feedback.get("completed_tests",feedback.get("tests",[])),
                    "pending_tests":feedback.get("pending_tests",[]),"original_task_pass_claimed":False})
                write_json_atomic(design.parent/"result.json",result)
                (design.parent/"completion-summary.md").write_text(
                    "# Agent budget tradeoff\n\nStatus: INCOMPLETE_EVIDENCE. No whole-machine PASS.\n\n"
                    +record["budget_decision"]["reason"]+"\n\nTradeoff: "+record["budget_decision"]["tradeoff"]
                    +"\n\nSee result.json and preserved prior revisions for per-test evidence.\n",encoding="utf-8")
                self.publish("incomplete",stop_reason="agent_chose_partial_delivery",active_turn={},active_test=None)
                return result
            if action=="adjust_task":
                if self.template_mode:
                    adjustment=design/"template_adjustment.json"
                    if not adjustment.is_file():return {"status":"NEEDS_REVISION","issues":["Write the requested template parameter adjustment"]}
                    return None
                try:
                    review=self.turn("task_adjustment_review",lambda directory:review_and_apply_adjustment(
                        self.reviewer,self.workspace,self.project,design,directory,context,
                        author=self.author,tier_name=self.tier.name if self.tier else "light",budget=self.budget))
                    self.state["author_session_id"]=self.author.session_id
                except (ValueError,FileNotFoundError) as exc:
                    review={"status":"needs_revision","applied":False,"summary":str(exc)}
                if review.get("status")=="provider_error":return review
                if not review.get("applied"):
                    feedback={"status":"TASK_ADJUSTMENT_NOT_ACCEPTED_BY_AUTHOR","issues":[review],"previous_feedback":feedback}
                    attempt+=1
                    continue
                self.criteria=read(self.project/"criteria.json")
                self.criteria_sha256=digest(self.project/"criteria.json")
                self.state.setdefault("task_adjustments",[]).append({
                    "revision":self.state["revision"],"history":review["adjustment_history_dir"],
                    "criteria_sha256":self.criteria_sha256,"budget_decision":str(design.parent/f"budget-decision-{attempt:02d}.json")})
                self.publish("geometry",active_turn={})
                response={"status":"ready"}
            if response.get("status")!="ready":
                self.publish("author_followup",stop_reason=response.get("status"),active_turn={})
                return response
            return None

    def run(self,*,one_revision=False):
        from .budget import ActiveBudgetExhausted
        from .construction_budget import BudgetExhausted
        from .test_coverage import FirstPassReserveReached
        with self.budget:
            try:
                try:
                    result=self._run(one_revision=one_revision)
                except FirstPassReserveReached as exc:
                    if exc.session_id:self.state["author_session_id"]=exc.session_id
                    self.publish("first_pass",first_pass_handoff=True,stop_reason="",active_turn={},active_test=None)
                    try:
                        result=self._run(one_revision=True)
                    except (FirstPassReserveReached,DeliveryError,OSError,ValueError) as blocked:
                        result={"status":"INCOMPLETE_EVIDENCE","issues":[str(exc),str(blocked)],
                                "first_pass_blocked":True,"complete_instance_pass":False}
                        self.publish("first_pass_blocked",stop_reason="no_executable_scene",active_turn={},active_test=None)
            except ActiveBudgetExhausted as exc:
                result=self.budget.incomplete(issues=[str(exc)])
                self.state=read(self.project/"workflow.json")
            except BudgetExhausted as exc:
                result=self.budget.incomplete(issues=[str(exc)]) if self.building else {"status":"BUDGET_EXHAUSTED","issues":[str(exc)]}
                self.state=read(self.project/"workflow.json")
            except BaseException as exc:
                self.publish("interrupted",interrupted_phase=self.state.get("phase"),stop_reason=type(exc).__name__,active_turn={},active_test=None)
                if self.construction:
                    from .construction import summary
                    try:
                        summary(self.project,reason=type(exc).__name__)
                    except Exception:
                        pass
                raise
            result=self._annotate_budget(result)
            if result.get("first_pass_blocked"):
                result.update(schema_version="slim_harness_result_v1",revision=int(self.state["revision"]))
                write_json_atomic(self.project/"runs"/f"r{int(self.state['revision']):02d}"/"result.json",result)
            if self.construction:
                from .construction import summary
                summary(self.project,reason=result.get("status",""),result=result)
            return result

    def _run(self,*,one_revision=False):
        if getattr(self,"_checkpoint_only",False):
            if not one_revision or self.state.get("phase") in {"author","author_followup"}:
                raise ValueError("Checkpoint evaluation cannot enter the author or repair loop")
        from .construction import author_turn_timeout
        from .publication import read_result
        existing=read_result(self.project/"runs"/f"r{int(self.state['revision']):02d}")
        if existing.get("eligible_pass") and (self.template_mode or self._numerical_cache_matches(existing.get("tests",[]),
                read(self.project/"runs"/f"r{int(self.state['revision']):02d}"/"design"/"scene.json")["tests"])):
            return existing
        assert_inactive(self.project)
        self.state["complete_instance_pass"]=False
        self.state["stop_reason"]=""
        with RuntimeHeartbeat(self.project,self.state):
            recovery_phase=self.state.get("interrupted_phase") if self.state.get("phase")=="interrupted" else self.state.get("phase")
            if recovery_phase=="recovery_wait":recovery_phase=self.state.get("recovery_from_phase")
            if recovery_phase in {"author","author_followup"}:
                if getattr(self,"_checkpoint_only",False):
                    raise ValueError("Checkpoint evaluation cannot resume an author turn")
                design=self.project/"runs"/f"r{int(self.state['revision']):02d}"/"design"
                recovery_feedback=("Read the previous revision's result.json and the latest worker_turns prompt for concrete feedback. "
                    if self.feedback_mode=="full"
                    else "The latest worker_turns prompt contains the available feedback. ")
                response=self.turn("author",lambda directory:self.author.run_turn(design,
                    f"Continue the existing machine in {design}; this is recovery of an interrupted author turn. "
                    "Preserve completed sources and prior revisions. "+recovery_feedback+
                    "Do not change the task, frozen criteria "
                    "or Harness. Deliver scene.json and sources using the Slim interface. "+author_instructions(self.project),
                    directory,author_turn_timeout(self.project)))
                self.state["author_session_id"]=self.author.session_id
                if response["status"]!="ready":
                    self.publish("author_followup",stop_reason=response["status"],active_turn={})
                    return response
            while True:
                self.budget.require_available()
                if self.building:self.construction.check_available()
                try: result=self.run_revision()
                except ActiveBudgetExhausted:
                    raise
                except ConstructionLimitError as exc:
                    result={"status":"NEEDS_REVISION","issues":[exc.details]}
                except ProviderRecoveryRequired as exc:
                    result=exc.result
                except DeliveryError as exc:
                    result={"status":"HARNESS_ERROR" if str(exc).startswith("CAPABILITY_GAP:") else "NEEDS_REVISION","issues":[str(exc)]}
                except Exception as exc:
                    import traceback
                    result={"status":"HARNESS_ERROR","issues":[f"{type(exc).__name__}: {exc}"],"traceback":traceback.format_exc()}
                    if self.budget_tracker is not None:
                        self.budget_tracker.pause("exception_recovery")
                        self.budget_tracker.save_checkpoint(self.project/"budget_checkpoint.json")
                result=self._annotate_budget(result)
                result.update(schema_version="slim_harness_result_v1",revision=int(self.state["revision"]),updated_at=datetime.now(timezone.utc).isoformat())
                run=self.project/"runs"/f"r{int(self.state['revision']):02d}"
                write_json_atomic(run/"result.json",result)
                if self.building:
                    from .construction import revision_choices
                    result=revision_choices(self.project,result)
                    write_json_atomic(run/"result.json",result)
                    if "REVIEW_PROVIDER_ERROR" in str(result.get("issues",[])):
                        self.publish("provider_error",stop_reason="review_provider_unavailable",active_test=None)
                        return result
                # Construction authors choose the repair route, and no problem in
                # agent-delivered work ends a budgeted autonomous loop: the author
                # receives it as revision evidence and the next revision gets a fresh
                # review. A run without any budget stops, as nothing would end it.
                # Do not invent PASS or drop already durable development evidence.
                returned=(self.building and not result.get("template_binding_error")) or (
                    not one_revision and (self.building or self.budget_tracker is not None))
                if result["status"] in {"HARNESS_ERROR","EVIDENCE_REQUIRED"} and returned:
                    result.update(status="NEEDS_REVISION",reported_status=result["status"])
                    write_json_atomic(run/"result.json",result)
                if result["status"]=="PASS":
                    self.publish("complete",complete_instance_pass=True,stop_reason="completed",active_test=None)
                    return result
                if result["status"] in {"ACTIVE_BUDGET_EXHAUSTED","BUDGET_EXHAUSTED"} or (result["status"]=="SIMULATION_TIMEOUT" and (one_revision or (self.budget_tracker is None and not self.building))):
                    if result["status"]=="ACTIVE_BUDGET_EXHAUSTED":
                        result=self.budget.incomplete(issues=["Active budget exhausted; available evidence retained"])
                        self.state=read(self.project/"workflow.json")
                    self.publish("budget_exhausted" if result["status"] in {"ACTIVE_BUDGET_EXHAUSTED","BUDGET_EXHAUSTED"} else "simulation_timeout",
                                 stop_reason=result["status"],active_test=None)
                    return result
                if result["status"]=="provider_error":
                    self.publish("provider_error",stop_reason="provider_error",active_turn={},active_test=None)
                    return result
                if result["status"]=="RESOURCE_POLICY_REQUIRED":
                    self.publish("resource_policy_required",stop_reason="RESOURCE_POLICY_REQUIRED",active_turn={},active_test=None)
                    return result
                if result["status"] in {"HARNESS_ERROR","EVIDENCE_REQUIRED"} or one_revision:
                    # The supervising goal owns concrete Harness repairs or new
                    # evidence work. No human approval state is introduced.
                    self.publish("evidence_collection" if result["status"]=="EVIDENCE_REQUIRED" else "harness_repair" if result["status"]=="HARNESS_ERROR" else "revision_needed",
                                 stop_reason=result["status"],active_test=None)
                    return result
                previous=run/"design"
                revision=int(self.state["revision"])+1
                design=self.project/"runs"/f"r{revision:02d}"/"design"
                if design.exists(): raise RuntimeError("Next revision already exists; preserve it")
                shutil.copytree(previous,design,ignore=shutil.ignore_patterns("__pycache__",".presim_gate"))
                self.publish("author",revision=revision,active_test=None)
                from .task_adaptation import adjustment_allowed
                if self.template_mode or self.budget_tracker is not None or adjustment_allowed(self.state):
                    outcome=self._budget_revision(design,result)
                    if outcome is not None:return outcome
                    continue
                from .feedback_ablation import transform_feedback
                prompt=(f"Continue the same machine in the new revision {design}. The previous revision remains preserved. "
                        "Make the smallest corrections justified by the concrete evidence below. "
                        +author_instructions(self.project)+
                        "Keep detail and the original task. Do not rewrite unaffected geometry or invent new acceptance rules. "
                        "Do not modify Harness code, task.md, criteria, review, evidence or result files. "
                        "Use the current slim_harness/README.md interface; no legacy template/package contracts. "
                        "Geometry, static/motion previews, time and Token are program-owned. Return ready when scene.json "
                        f"and its sources are complete.\nFROZEN CRITERIA:\n{json.dumps(self.criteria,ensure_ascii=False)}\n"
                        f"CONCRETE FEEDBACK:\n{json.dumps(transform_feedback(result,self.feedback_mode),ensure_ascii=False)}")
                if self.building:
                    from .construction import policy_prompt
                    prompt+=policy_prompt(self.project)
                from .construction import author_turn_timeout
                response=self.turn("author",lambda directory:self.author.run_turn(design,prompt,directory,author_turn_timeout(self.project)))
                self.state["author_session_id"]=self.author.session_id
                if response["status"]!="ready":
                    self.publish("author_followup",stop_reason=response["status"])
                    return response


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path)
    parser.add_argument("--workspace",type=Path,default=Path.cwd())
    parser.add_argument("--codex",default="codex")
    parser.add_argument("--one-revision",action="store_true")
    parser.add_argument("--tier",choices=["light","medium","high"],default=None,
                        help="Budget tier: light (1h/1min/2tasks/10parts), medium (2h/3min/2tasks/15parts), high (3h/5min/5tasks/20parts)")
    parser.add_argument("--physics-timeout-seconds",type=float,default=600,
                        help="Wall budget for each native test process, including its numerical retries (default: 600)")
    parser.add_argument("--physics-max-step-seconds",type=float,default=.001,
                        help="Harness numerical step upper bound, positive and at most 0.001 seconds (default: 0.001)")
    parser.add_argument("--physics-solver-iterations",type=int,default=160,
                        help="Positive PSOR iteration budget; friction clutches retain their 240 minimum (default: 160)")
    parser.add_argument("--worker-provider",choices=list(WORKER_PROVIDERS),default="codex",
                        help="LLM worker backend: codex (Codex CLI), cursor (Cursor CLI), or anthropic")
    parser.add_argument("--model",default=None,help="Override the default model for the worker provider")
    parser.add_argument("--feedback-mode",choices=["full","binary"],default="full",
                        help="Feedback fidelity for ablation: full (default) or binary (test-level pass/fail only)")
    args=parser.parse_args(argv)
    from .construction import budget_for, construction_requested
    if budget_for(args.project) is not None and construction_requested(args.project):
        from .construction_supervisor import supervise_project
        result=supervise_project(args.workspace,args.project,executable=args.codex,one_revision=args.one_revision,
                                 feedback_mode=args.feedback_mode)
    else:
        result=SlimWorkflow(args.workspace,args.project,executable=args.codex,
                            model=args.model or default_worker_model(args.worker_provider),
                            physics_timeout_seconds=args.physics_timeout_seconds,
                            physics_max_step_seconds=args.physics_max_step_seconds,
                            physics_solver_iterations=args.physics_solver_iterations,
                            tier=args.tier,worker_provider=args.worker_provider,
                            feedback_mode=args.feedback_mode).run(one_revision=args.one_revision)
    print(json.dumps({k:v for k,v in result.items() if k not in {"tests","review"}},ensure_ascii=False))


if __name__=="__main__": main()
