"""Public create, submit, and resume commands; all workflow files are internal."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from ..cold_start.runtime_status import RuntimeHeartbeat
from ..cold_start.worker_factory import WORKER_PROVIDERS, create_worker, default_worker_model, normalize_worker_provider
from ..mechanism.io import write_json_atomic
from .contracts import load_scene
from .task_adaptation import initial_task_criteria
from .project_isolation import CODE_ROOT, ENV_VISUAL, harness_umask
from .frozen_experiment import install_experiment, load_experiment, author_instructions, validate_experiment
from .workflow import SlimWorkflow, digest, assert_inactive, ActiveWorkflowError
from .budget import recommended_limits
from .construction_budget import BUDGET_HOURS
from .run_hooks import after_run


REASONING_EFFORTS=("low","medium","high","xhigh","max","ultra")


def _project_reasoning(project,state,override):
    selected=override if override is not None else state["request"].get("reasoning_effort","medium")
    if selected not in REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {selected!r}")
    if override is not None and state["request"].get("reasoning_effort")!=selected:
        state["request"]["reasoning_effort"]=selected
        write_json_atomic(project/"workflow.json",state)
    return selected


def create_project(workspace: Path, project_id: str, task: str, title: str | None = None) -> Path:
    if not project_id or Path(project_id).name!=project_id or project_id in {".",".."}:
        raise ValueError("Use a single project name")
    if not task.strip(): raise ValueError("A natural-language task is required")
    project=Path(workspace).resolve()/".runtime_runs"/"projects"/project_id
    if project.exists(): raise ValueError("Project already exists; use resume to preserve its history")
    (project/"runs"/"r01"/"design").mkdir(parents=True)
    (project/"task.md").write_text(task,encoding="utf-8")
    now=datetime.now(timezone.utc).isoformat()
    write_json_atomic(project/"workflow.json",{
        "schema_version":"slim_harness_workflow_v1","request":{"project_id":project_id,"title":title or project_id,"brief":task},
        "revision":1,"phase":"criteria","started_at":now,"updated_at":now,
        "events":[],"complete_instance_pass":False,"elapsed_seconds":0.,
    })
    return project


def copy_delivery(scene_path: Path, design: Path, *, experiment=None, preserve_source_layout=False, model_only=False):
    from .contracts import load_model_scene
    scene=load_model_scene(scene_path) if model_only else load_scene(scene_path, experiment=experiment)
    names={}
    used_names={"scene.json"}
    for body in scene["bodies"]:
        source=(scene_path.parent/body["source"]).resolve()
        if source not in names:
            if preserve_source_layout:
                name=source.relative_to(scene_path.parent).as_posix()
                (design/name).parent.mkdir(parents=True,exist_ok=True)
            else:
                name=source.name
                index=len(names)+1
                while name.casefold() in used_names:
                    name=f"source_{index:03d}{source.suffix}"
                    index+=1
            shutil.copy2(source,design/name)
            names[source]=name
            used_names.add(name.casefold())
        body["source"]=names[source]
    # Modeling sources are optional. Copy only explicitly declared files,
    # never the whole directory containing a user's scene.
    authoring_sources=list(scene.get("authoring_sources",[]))
    if model_only:
        authoring_sources.extend(name for name in ("parts.json","all_print_inventory.json","print/inventory.json","task_template.py","task_template_dependencies.json","template_adjustment.json")
                                 if name not in authoring_sources and (scene_path.parent/name).is_file())
        dependency_file=scene_path.parent/"task_template_dependencies.json"
        if dependency_file.is_file():
            dependencies=json.loads(dependency_file.read_text(encoding="utf-8-sig"))
            if not isinstance(dependencies,list) or any(not isinstance(p,str) for p in dependencies):
                raise ValueError("task_template_dependencies.json must list relative dependency files")
            authoring_sources.extend(p for p in dependencies if p not in authoring_sources)
    for relative in authoring_sources:
        source=(scene_path.parent/relative).resolve()
        if Path(relative).is_absolute() or not source.is_relative_to(scene_path.parent):
            raise ValueError("Keep authoring dependencies below the scene directory")
        target=design/relative
        if target.parent==design and target.name in names.values(): raise ValueError("Authoring-source filename conflicts with geometry")
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    write_json_atomic(design/"scene.json",scene)


def _tier_name(tier):
    return getattr(tier,"name",tier)


@after_run
def start(workspace,project_id,task=None,*,title=None,scene_path=None,executable="codex",experiment_path=None,worker_provider="codex",model=None,tier=None,task_adjustable=False,reasoning=None,reasoning_effort=None,budget_config=None,chrono_python=None,construction=None,mode=None,reviewer_provider=None,reviewer_model=None,reviewer_executable=None,reviewer_reasoning=None,feedback_mode="full",author_visual_policy="full"):
    from ..benchmark_sandbox.visual_policy import validate as validate_visual_policy, WORKER_NO_VISUALS
    author_visual_policy=validate_visual_policy(author_visual_policy)
    if author_visual_policy == WORKER_NO_VISUALS and normalize_worker_provider(worker_provider) != "codex":
        raise ValueError("worker_no_visuals_v1 currently requires the original Codex author")
    from .execution import formal_sandbox_required
    if author_visual_policy == WORKER_NO_VISUALS and not formal_sandbox_required():
        # Only the formal sandbox installs the author's visual restrictions.
        raise ValueError("worker_no_visuals_v1 requires the formal sandbox; this installation runs locally")
    workspace=Path(workspace).resolve()
    from . import modes
    mode_configuration=modes.configure(mode) if mode else None
    if mode_configuration and (experiment_path or task_adjustable):
        raise ValueError("Free creation owns task adaptation; do not combine it with --experiment/--adjustable")
    task=modes.resolve_task(task)
    if reasoning_effort is not None:
        reasoning=reasoning_effort
    if reasoning is not None and reasoning not in REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {reasoning!r}")
    if task_adjustable and experiment_path is not None:
        raise ValueError("Frozen benchmark tasks cannot be adjustable")
    if task_adjustable and tier is None:
        raise ValueError("Adjustable task tradeoffs require an explicit budget tier")
    if experiment_path is not None:
        validate_experiment(json.loads(Path(experiment_path).read_text(encoding="utf-8-sig")),task)
    project=create_project(workspace,project_id,task,title)
    if mode_configuration:modes.install(project,mode_configuration)
    building=bool(construction) if construction is not None else (scene_path is None and experiment_path is None and _tier_name(tier) in (None,"high"))
    if building and experiment_path is not None:
        raise ValueError("Frozen experiments cannot use construction test-relaxation permissions")
    light_closeout=(str(_tier_name(tier) or "").lower()=="light" and experiment_path is None)
    if building or light_closeout:
        from .construction_budget import BudgetConfig, initialize_budget
        from .budget import get_tier
        selected=get_tier(_tier_name(tier) or "high")
        ledger=budget_config if budget_config is not None else (BudgetConfig(hours=1) if light_closeout and not building else
            BudgetConfig(hours=int(selected.active_budget_seconds/3600),simulation_timeout_seconds=selected.simulation_timeout_seconds,
                         max_tasks=selected.max_tasks_per_design,max_tests=selected.max_tasks_per_design))
        initialize_budget(project,ledger)
    state=json.loads((project/"workflow.json").read_text(encoding="utf-8"))
    state.update(stop_reason="",complete_instance_pass=False)
    design=project/"runs"/"r01"/"design"
    state["request"]["delivery_mode"]="submit" if scene_path is not None else "create"
    state["request"]["task_adjustable"]=bool(task_adjustable)
    state["request"]["construction"]=bool(building)
    if mode_configuration:state["request"]["mode"]=mode
    if building:state["request"]["tier"]=_tier_name(tier) or "high"
    worker_provider=normalize_worker_provider(worker_provider)
    effective_model=model or state["request"].get("model") or default_worker_model(worker_provider)
    default_reasoning="high" if worker_provider=="codex" else "medium"
    state["request"].update(model=effective_model,worker_provider=worker_provider,
                            reasoning_effort=reasoning if reasoning is not None else default_reasoning,
                            chrono_python=chrono_python,feedback_mode=feedback_mode,
                            author_visual_policy=author_visual_policy)
    if worker_provider in {"codex","cursor"}:
        reviewer_provider=normalize_worker_provider(reviewer_provider) if reviewer_provider is not None else "codex"
        if reviewer_model is None:reviewer_model=default_worker_model(reviewer_provider)
        if reviewer_reasoning is None:reviewer_reasoning="medium"
    for key,value in {"reviewer_provider":reviewer_provider,"reviewer_model":reviewer_model,"reviewer_executable":reviewer_executable,"reviewer_reasoning_effort":reviewer_reasoning}.items():
        if value is not None:state["request"][key]=value
    if experiment_path is not None:
        state["request"]["frozen_experiment_sha256"]=install_experiment(project,Path(experiment_path))
    write_json_atomic(project/"workflow.json",state)
    if scene_path is not None: copy_delivery(Path(scene_path).resolve(),design,experiment=load_experiment(project),model_only=bool(mode_configuration),preserve_source_layout=bool(mode_configuration))
    if building:
        from .construction_supervisor import supervise_project
        return supervise_project(workspace,project,executable=executable,feedback_mode=feedback_mode)
    return initialize_project(workspace,project,executable=executable,worker_provider=worker_provider,model=model,tier=tier,
                              reasoning=state["request"]["reasoning_effort"],feedback_mode=feedback_mode)


def initialize_project(workspace,project,*,executable="codex",worker_provider="codex",model=None,tier=None,reasoning=None,feedback_mode="full"):
    assert_inactive(project)
    state=json.loads((project/"workflow.json").read_text(encoding="utf-8"))
    from .construction import recover_author_session
    state=recover_author_session(project,state)
    write_json_atomic(project/"workflow.json",state)
    effective_reasoning=_project_reasoning(project,state,reasoning)
    from .budget import ProjectBudget, ActiveBudgetExhausted
    from .test_coverage import FirstPassReserveReached
    with ProjectBudget(project, tier) as budget:
        try:
            budget.require_available()
            result = _initialize_project(workspace, project, executable=executable,
                worker_provider=worker_provider, model=model, budget=budget, reasoning=effective_reasoning)
        except FirstPassReserveReached as exc:
            state=json.loads((project/"workflow.json").read_text(encoding="utf-8"))
            if exc.session_id:state["author_session_id"]=exc.session_id
            design=project/"runs"/f"r{int(state['revision']):02d}"/"design"/"scene.json"
            try:
                from .modes import settings as mode_settings
                if mode_settings(project):
                    from .contracts import load_model_scene
                    load_model_scene(design)
                else:load_scene(design,experiment=load_experiment(project))
                usable=(project/"criteria.lock.json").is_file()
            except (OSError,ValueError,KeyError,TypeError):
                usable=False
            if usable:
                state.update(first_pass_handoff=True,phase="first_pass",stop_reason="",active_turn={})
                state["events"].append({"kind":"first_pass_handoff","at":datetime.now(timezone.utc).isoformat(),
                    "result":{"status":"partial_author_delivery","summary":str(exc)}})
                write_json_atomic(project/"workflow.json",state)
                result=None
            else:
                from .test_coverage import planned_tests,coverage_summary
                tests=planned_tests(project) or [];pending=[test["id"] for test in tests]
                result={"status":"INCOMPLETE_EVIDENCE","issues":[str(exc),"No executable scene was available for the reserved first test pass"],
                        "first_pass_blocked":True,"complete_instance_pass":False,"pending_tests":pending,"unrun_tests":pending,
                        "test_coverage":coverage_summary(tests,[],pending)}
                state.update(phase="first_pass_blocked",stop_reason="no_executable_scene",active_turn={})
                write_json_atomic(project/"workflow.json",state)
        except ActiveBudgetExhausted as exc:
            result = budget.incomplete(issues=[str(exc)])
        except BaseException as exc:
            state=json.loads((project/"workflow.json").read_text(encoding="utf-8"))
            state.update(phase="interrupted",stop_reason=type(exc).__name__,active_turn={})
            write_json_atomic(project/"workflow.json",state)
            from .construction import budget_for,summary
            if budget_for(project) is not None:
                try:
                    summary(project,reason=type(exc).__name__)
                except Exception:
                    pass
            raise
        if result is not None:
            annotated=budget.annotate(result)
            if result.get("first_pass_blocked"):
                annotated.update(schema_version="slim_harness_result_v1",revision=int(state["revision"]))
                write_json_atomic(project/"runs"/f"r{int(state['revision']):02d}"/"result.json",annotated)
            from .construction import budget_for,summary
            if budget_for(project) is not None:
                summary(project,reason=annotated.get("status",""),result=annotated)
            return annotated
    state=json.loads((project/"workflow.json").read_text(encoding="utf-8"))
    worker_provider=normalize_worker_provider(state["request"].get("worker_provider",worker_provider))
    effective_model=model or state["request"].get("model") or default_worker_model(worker_provider)
    feedback_mode=state["request"].get("feedback_mode",feedback_mode)
    return SlimWorkflow(workspace,project,executable=executable,model=effective_model,
                        worker_provider=worker_provider,tier=tier,reasoning=effective_reasoning,
                        feedback_mode=feedback_mode).run(
                            one_revision=state["request"].get("delivery_mode")=="submit")


def _initialize_project(workspace,project,*,executable,worker_provider,model,budget,reasoning="medium"):
    state=json.loads((project/"workflow.json").read_text(encoding="utf-8"))
    budget.state=state
    state.update(stop_reason="",complete_instance_pass=False)
    design=project/"runs"/"r01"/"design"
    task=(project/"task.md").read_text(encoding="utf-8")
    submitted=state["request"].get("delivery_mode")=="submit"
    experiment=load_experiment(project)
    from .modes import settings as mode_settings,author_instructions as mode_instructions
    worker_provider=normalize_worker_provider(state["request"].get("worker_provider",worker_provider))
    effective_model=model or state["request"].get("model") or default_worker_model(worker_provider)
    from .construction import worker_for,policy_prompt,author_turn_timeout,construction_requested
    author=create_worker(provider=worker_provider,model=effective_model,reasoning_effort=reasoning,executable=executable,
                          session_id=state.get("author_session_id"),disabled_features=("multi_agent","multi_agent_v2"))
    from .sandbox_binding import bind_worker
    author=bind_worker(author,workspace,project,'author',state.get('request',{}).get('author_visual_policy','full'))
    author=budget.worker(author)
    author=worker_for(project,author)
    def next_turn(kind):
        prior=list((project/"worker_turns").glob("t*_*"))
        index=max([int(p.name.split("_")[0][1:]) for p in prior]+[0])+1
        directory=project/"worker_turns"/f"t{index:03d}_{kind}"
        directory.mkdir(parents=True)
        return directory
    with RuntimeHeartbeat(project,state):
        if not (project/"criteria.json").is_file():
            criteria_turn=next_turn("task_binding")
            state.update(phase="criteria",stop_reason="",active_turn={})
            if experiment is None:
                result={"status":"ready","criteria":initial_task_criteria(task),
                        "summary":"Original task bound by code; no initial model review.","source":"program_task_binding"}
            else:
                result={"status":"ready","criteria":experiment["criteria"],"summary":experiment["reviewer_summary"],"source":"frozen_experiment"}
            write_json_atomic(project/"criteria.json",{
                "schema_version":"slim_frozen_criteria_v1","task_file":str(project/"task.md"),
                "task_text":task,"criteria":result["criteria"],"reviewer_summary":result["summary"],"source":result["source"]})
            write_json_atomic(criteria_turn/"task-binding.json",result)
            state["events"].append({"kind":"criteria","at":datetime.now(timezone.utc).isoformat(),"result":{**result,"turn_directory":str(criteria_turn)}})
        task=(project/"task.md").read_text(encoding="utf-8")
        lock_path=project/"criteria.lock.json"
        if lock_path.is_file():
            if json.loads(lock_path.read_text(encoding="utf-8"))["criteria_sha256"]!=digest(project/"criteria.json"):
                raise ValueError("Frozen criteria changed; a sourced correction is required")
        else:
            frozen=json.loads((project/"criteria.json").read_text(encoding="utf-8"))
            original=next((e["result"] for e in state.get("events",[]) if e.get("kind")=="criteria" and e.get("result",{}).get("status")=="ready"),{})
            if frozen["criteria"]!=original.get("criteria") or frozen["reviewer_summary"]!=original.get("summary"):
                raise ValueError("Criteria differ from their recorded task binding")
            write_json_atomic(lock_path,{"schema_version":"slim_criteria_lock_v1",
                              "criteria_sha256":digest(project/"criteria.json"),"task_sha256":digest(project/"task.md")})
        if not submitted:
            turn=next_turn("author")
            state.update(phase="author",active_turn={"id":turn.name,"directory":str(turn)})
            write_json_atomic(project/"workflow.json",state)
            prompt=(f"Build the machine in {design}. Use any available modeling tool and preserve detailed, recognizable geometry. "
                    f"Read {CODE_ROOT/'CAD_TOOLCHAIN.md'} and {CODE_ROOT/'simulation_ready_cad_workflow/validation_pipeline/slim_harness/README.md'}. "
                    "This is the slim scene interface: deliver scene.json and source geometry. "
                    +author_instructions(project)+
                    "No legacy project/package/template/role contracts are required. The program owns collider generation, "
                    "physics, images, motion, time and Token; do not write verdicts or update the dashboard. "
                    +("Define meaningful initial simulation tasks yourself. There is no initial, preflight or final model review. "
                      "Only later changes to simulation tasks receive time-limited advice; you decide from that advice while all code gates remain binding. "
                      if not mode_settings(project) else mode_instructions(project))+
                    "Use the fixed CAD Python if authoring CAD. Do not modify task, criteria, or Harness code. "
                    "Geometry starts in assembled world coordinates. Preserve failed source revisions and return ready "
                    "when the delivery is complete. Report a missing physical capability rather than claiming unsupported evidence.\n"
                    +(policy_prompt(project) if construction_requested(project) else '')+
                    mode_instructions(project)+
                    ('\nPrevious incomplete turn: '+json.dumps(state['last_construction_feedback'],ensure_ascii=False)+'\nChoose a different supported construction strategy now.\n' if state.get('last_construction_feedback') else '')+
                    f"TASK:\n{task}\n"
                    f"FROZEN CRITERIA:\n{(project/'criteria.json').read_text(encoding='utf-8')}")
            result=author.run_turn(design,prompt,turn,author_turn_timeout(project))
            state["events"].append({"kind":"author","at":datetime.now(timezone.utc).isoformat(),"result":{**result,"turn_directory":str(turn)}})
            state["author_session_id"]=author.session_id
            if result["status"]=="ready":state.pop("first_pass_handoff",None)
            if result["status"]!="ready":
                if construction_requested(project):
                    write_json_atomic(project/"workflow.json",state)
                    author_turn_timeout(project)  # raises at the first-pass reserve
                state.update(phase="author_followup",stop_reason=result["status"])
                write_json_atomic(project/"workflow.json",state)
                return result
        state.update(phase="geometry",active_turn={},updated_at=datetime.now(timezone.utc).isoformat())
        write_json_atomic(project/"workflow.json",state)
    return None


@after_run
def resume_project(workspace,project,*,executable="codex",worker_provider="codex",model=None,tier=None,reasoning=None,in_process=False,feedback_mode=None):
    from .construction import budget_for
    state=json.loads((Path(project)/"workflow.json").read_text(encoding="utf-8"))
    if budget_for(project) is not None and not in_process and state.get("request",{}).get("construction"):
        from .construction_supervisor import supervise_project
        return supervise_project(workspace,project,executable=executable,feedback_mode=feedback_mode or state.get("request",{}).get("feedback_mode","full"))
    assert_inactive(project)
    state=json.loads((project/"workflow.json").read_text(encoding="utf-8"))
    effective_reasoning=_project_reasoning(project,state,reasoning)
    has_author=bool(state.get("first_pass_handoff")) or any(event.get("kind")=="author" and event.get("result",{}).get("status")=="ready" for event in state.get("events",[]))
    submitted=state["request"].get("delivery_mode")=="submit"
    effective_feedback_mode=feedback_mode or state.get("request",{}).get("feedback_mode","full")
    if not (project/"criteria.json").is_file() or (not submitted and not has_author):
        return initialize_project(workspace,project,executable=executable,worker_provider=worker_provider,model=model,tier=tier,reasoning=reasoning,feedback_mode=effective_feedback_mode)
    worker_provider=normalize_worker_provider(state["request"].get("worker_provider",worker_provider))
    effective_model=model or state["request"].get("model") or default_worker_model(worker_provider)
    return SlimWorkflow(workspace,project,executable=executable,model=effective_model,worker_provider=worker_provider,tier=tier,
                        reasoning=effective_reasoning,feedback_mode=effective_feedback_mode).run(one_revision=submitted)


def main(argv=None):
    parser=argparse.ArgumentParser(description="Submit a mechanical scene or build a machine from a natural-language task")
    parser.add_argument("--workspace",type=Path,default=Path.cwd())
    parser.add_argument("--codex",default="codex")
    parser.add_argument("--worker-provider",choices=list(WORKER_PROVIDERS),default="codex",
                        help="LLM worker backend: codex (Codex CLI), cursor (Cursor CLI), claude_code (Claude Code CLI), or anthropic")
    parser.add_argument("--model",default=None,help="Override the default model for the worker provider")
    parser.add_argument("--reasoning",choices=REASONING_EFFORTS,default=None,
                        help="Worker reasoning effort; new Codex projects default to high, resume restores the saved selection")
    parser.add_argument("--tier",choices=["light","medium","high"],default=None,
                        help="New create/submit defaults to high (5 tests, 20 parts); light (2 tests, 10 parts) or medium (2 tests, 15 parts); "
                             "resume preserves its saved tier. Recommended --budget-hours / --simulation-timeout: "
                             +", ".join(f"{name} {recommended_limits(name)[0]} h / {recommended_limits(name)[1]:g} s" for name in ("light","medium","high")))
    parser.add_argument("--feedback-mode",choices=["full","binary"],default="full",
                        help="Feedback fidelity for ablation: full (default) or binary (test-level pass/fail only)")
    parser.add_argument("--author-visual-policy",choices=["full","worker_no_visuals_v1"],default=None,
                        help="Author visual feedback policy; worker_no_visuals_v1 disables supported render/view paths")
    sub=parser.add_subparsers(dest="command",required=True)
    for verb in ("create","submit"):
        command=sub.add_parser(verb)
        if verb=="submit": command.add_argument("scene",type=Path)
        command.add_argument("--project",required=True)
        command.add_argument("--title")
        command.add_argument("--mode",choices=("free",),help="Construction mode; free creation is the only mode and the default for create/submit")
        command.add_argument("--reviewer-provider",choices=list(WORKER_PROVIDERS))
        command.add_argument("--reviewer-model")
        command.add_argument("--reviewer-executable")
        command.add_argument("--reviewer-reasoning",choices=REASONING_EFFORTS)
        command.add_argument("--experiment",type=Path,help="Researcher-frozen Slim criteria and tests (JSON); omitted for automatic test design")
        command.add_argument("--adjustable",action="store_true",help="Allow author-decided task tradeoffs with time-limited change-review advice; incompatible with frozen benchmarks")
        high_hours,high_seconds=recommended_limits("high")
        command.add_argument("--budget-hours",type=int,choices=BUDGET_HOURS,default=None,
                             help=f"Total active construction time in hours; default: the tier's recommendation (high: {high_hours})")
        command.add_argument("--simulation-timeout",type=float,default=None,
                             help=f"Wall-time cap for one simulation, in seconds; default: the tier's recommendation (high: {high_seconds:g})")
        command.add_argument("--audit-timeout",type=float,default=None,
                             help="Construction-only source-motion audit wall cap; separate from native simulation timeout")
        command.add_argument("--max-tests",type=int,default=None)
        command.add_argument("--max-tasks",type=int,default=None)
        command.add_argument("--max-geometry-units",type=int,default=None,
                             help="Construction-only total source/explicit decomposition unit cap")
        command.add_argument("--max-geometry-units-per-body",type=int,default=None,
                             help="Construction-only per-rigid-body geometry unit cap")
        command.add_argument("--reasoning-effort",default=None,choices=REASONING_EFFORTS)
        command.add_argument("--chrono-python",default=None)
        command.add_argument("--construction",action="store_true",default=None,help="Use supervised construction for submitted designs; new create already uses it")
        inputs=command.add_mutually_exclusive_group(required=False)
        inputs.add_argument("--task",type=Path)
        inputs.add_argument("--brief")
    resume=sub.add_parser("resume")
    resume.add_argument("project")
    budget_parser=sub.add_parser("budget",help="Inspect or explicitly pause/resume effective construction time")
    budget_parser.add_argument("action",choices=("status","pause","resume","reclassify"))
    budget_parser.add_argument("--project",required=True)
    budget_parser.add_argument("--reason",choices=("environment_setup","human_intervention","provider_wait","stalled_wait"))
    budget_parser.add_argument("--seconds",type=float,default=None,
                              help="For reclassify: move this many already-charged active seconds into --reason")
    complete=sub.add_parser("completion",help="Write evidence-based partial completion summary")
    complete.add_argument("--project",required=True)
    complete.add_argument("--reason",default="requested_summary")
    execute=sub.add_parser("_construction-execute",help=argparse.SUPPRESS)
    execute.add_argument("--project",required=True)
    execute.add_argument("--one-revision",action="store_true",help=argparse.SUPPRESS)
    execute.add_argument("--feedback-mode",choices=["full","binary"],default="full",help=argparse.SUPPRESS)
    audit=sub.add_parser("audit",help="Audit an existing saved trajectory; no physics rerun or final publication")
    audit.add_argument("request",type=Path,help="Program-owned simulation request.json")
    audit.add_argument("--trace",type=Path,help="Saved trace or prefix.json; defaults to request.output")
    audit.add_argument("--output",type=Path,help="New geometry-audit.json output; preserves existing reports")
    for verb in ("build","preview","check","simulate","probe"):
        command=sub.add_parser(verb,help="Draft tool; does not publish a final mechanical verdict")
        command.add_argument("source",type=Path,help="model.py returning Asset, or scene.json (except build)")
        command.add_argument("--project",help="Existing Slim project; inferred for its own design files")
        if verb=="build":command.add_argument("--output",type=Path)
        if verb in {"simulate","probe"}:
            command.add_argument("--test",help="Run one complete declared test; never shorten a frozen test")
            command.add_argument("--chrono-python",default=None)
    args=parser.parse_args(argv)
    harness_umask(args.workspace)
    if args.author_visual_policy is not None and args.command not in {"create", "submit", "resume"}:
        parser.error("--author-visual-policy belongs to create/submit; resume preserves the saved policy")
    if args.command == "resume" and args.author_visual_policy is not None:
        saved=json.loads((args.workspace/".runtime_runs/projects"/args.project/"workflow.json").read_text(encoding="utf-8"))
        if args.author_visual_policy != saved.get("request",{}).get("author_visual_policy","full"):
            parser.error("Resume cannot change the author visual policy; create a fresh comparison project")
    if args.command in {"budget","completion","_construction-execute"}:
        project=args.workspace.resolve()/".runtime_runs"/"projects"/args.project
        from .construction import budget_for,summary
        budget=budget_for(project)
        if budget is None:raise ValueError("Not a budgeted construction project")
        if args.command=="completion":result=summary(project,reason=args.reason)
        elif args.command=="_construction-execute":
            from .construction_supervisor import execute_project
            result=execute_project(args.workspace,project,executable=args.codex,one_revision=args.one_revision,feedback_mode=args.feedback_mode)
        elif args.action=="status":result=budget.snapshot()
        elif args.action=="pause":
            if not args.reason:raise ValueError("A concrete exclusion --reason is required")
            result=budget.pause(args.reason)
        elif args.action=="reclassify":
            if not args.reason:raise ValueError("A concrete exclusion --reason is required")
            if args.seconds is None:raise ValueError("reclassify requires --seconds")
            result=budget.reclassify_active(args.reason,args.seconds)
        else:result=budget.resume()
        if args.command=="budget":
            from .construction import review_notes
            notes=review_notes(project)
            if notes:result["construction_review_notes"]=notes
        print(json.dumps(result,ensure_ascii=False));return 0
    if args.command=="audit":
        from .audit import audit_saved_request
        output=args.output or args.workspace/".runtime_runs"/"source_audits"/datetime.now(timezone.utc).strftime("audit_%Y%m%dT%H%M%S_%f.json")
        result=audit_saved_request(args.request,output,trace_path=args.trace)
        print(json.dumps({"geometry_passed":result["passed"],"audit":str(output),"certified":False,
            "available_solver_samples":result["available_solver_samples"],"checked_solver_samples":result["checked_solver_samples"],
            "sampling_policy":result["sampling_policy"],"violations":result["violations"]},ensure_ascii=False))
        return 0 if result["passed"] else 2
    elif args.command in {"build","preview","check","simulate","probe"}:
        from .author_tools import run_tool, CHRONO_PYTHON
        result=run_tool(args.command,args.source,workspace=args.workspace,project=args.project,
            output=getattr(args,"output",None),test_id=getattr(args,"test",None),
            chrono_python=getattr(args,"chrono_python",None) or CHRONO_PYTHON,
            visual_policy=os.environ.get(ENV_VISUAL,"full"))
        compact={k:result[k] for k in ("status","certified","scope","attempt","scene","image","model",
            "geometry","initial_audit","issues","available_tests","elapsed_seconds","construction_budget","revision_strategies","required_action","construction_review_notes","cost_probe") if k in result}
        if "tests" in result:
            compact["tests"]=[({"id":t["test_id"],"physical_pass":t["physical_pass"],
                "geometry_pass":t["geometry_audit"]["passed"],"failure":t["failure"],
                "failed_assertions":[a for a in t["assertions"] if not a.get("passed")],
                "trace_file":t["trace_file"]} if isinstance(t,dict) else t) for t in result["tests"]]
        print(json.dumps(compact,ensure_ascii=False))
        return 0 if result["status"] in {"BUILT","PREVIEW_READY","CHECKED","DEVELOPMENT_PHYSICS_PASSED","COST_PROBE_COMPLETE"} else 2
    elif args.command=="resume":
        project=args.workspace/".runtime_runs"/"projects"/args.project
        try: result=resume_project(args.workspace,project,executable=args.codex,worker_provider=args.worker_provider,model=args.model,tier=args.tier,reasoning=args.reasoning,feedback_mode=args.feedback_mode)
        except ActiveWorkflowError as exc: result={"status":"RUNNING","issues":[str(exc)]}
    else:
        mode=args.mode or (None if args.experiment is not None else "free")
        task=args.brief if args.brief is not None else args.task.read_text(encoding="utf-8") if args.task else None
        if task is None:
            parser.error("Provide --task or --brief for a new project")
        from .construction_budget import BudgetConfig
        from .budget import get_tier
        selected_tier=args.tier or "high"
        selected=get_tier(selected_tier)
        hours,simulation_seconds=recommended_limits(selected_tier)
        config=BudgetConfig(hours=args.budget_hours or hours,
            simulation_timeout_seconds=args.simulation_timeout if args.simulation_timeout is not None else simulation_seconds,
            max_tests=args.max_tests or selected.max_tasks_per_design,max_tasks=args.max_tasks or selected.max_tasks_per_design,
            max_geometry_units=args.max_geometry_units or 2048,
            max_geometry_units_per_body=args.max_geometry_units_per_body or 128,
            audit_timeout_seconds=args.audit_timeout if args.audit_timeout is not None else 600)
        result=start(args.workspace,args.project,task,title=args.title,scene_path=getattr(args,"scene",None),executable=args.codex,experiment_path=args.experiment,worker_provider=args.worker_provider,model=args.model,tier=selected_tier,task_adjustable=args.adjustable,reasoning=args.reasoning,reasoning_effort=args.reasoning_effort,budget_config=config,chrono_python=args.chrono_python,construction=args.construction,mode=mode,reviewer_provider=args.reviewer_provider,reviewer_model=args.reviewer_model,reviewer_executable=args.reviewer_executable,reviewer_reasoning=args.reviewer_reasoning,feedback_mode=args.feedback_mode,author_visual_policy=args.author_visual_policy)
    print(json.dumps({"status":result.get("status"),"issues":result.get("issues",[]),"attempt":result.get("attempt"),
                      "budget_timing":result.get("budget_timing"),"pending_tests":result.get("pending_tests")},ensure_ascii=False))
    return 0 if result.get("status") in {"PASS","READY","ready","RUNNING","ACTIVE_BUDGET_EXHAUSTED","BUDGET_EXHAUSTED"} else 2
