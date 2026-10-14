"""One scoped physics launcher shared by formal and development execution."""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import BoundedSubprocessError, run_bounded_subprocess
from ..runtime.system_probe import SystemProbe
from .budget import ActiveBudgetExhausted, ObservedWaitRecoveryRequired
from .resource_wait import load_declared_resource_requirement, wait_for_observed_resources, resource_policy_required_result


def retained_prefixes(folder):
    from .trace_storage import open_trace_prefix, release_trace_cache
    readable=[];unreadable=[]
    for path in Path(folder).rglob("prefix.json"):
        try:
            trace=open_trace_prefix(path)
            readable.append({"path":str(path),"confirmed_samples":len(trace["samples"]),"status":"UNVERIFIED_PREFIX"})
            release_trace_cache(trace)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            unreadable.append({"path":str(path),"reason":str(exc)})
    return {"readable_prefixes":readable,"unreadable_prefixes":unreadable,
            "prefix_status":"UNVERIFIED_PREFIX" if readable else "NO_READABLE_PREFIX",
            "recovery_scope":"Retain failed attempts; restart the failed frozen test from its initial state. No solver continuation or PASS is claimed."}


def _rss_limit(project, modeling_policy):
    """Keep explicit resource settings; host admission and per-process RSS differ."""
    sources=[modeling_policy or {}]
    if project:
        for name in (("modeling_policy.json",) if modeling_policy is None else ())+("scope.json","resource_policy.json"):
            path=Path(project)/name
            if path.is_file():sources.append(json.loads(path.read_text(encoding="utf-8-sig")))
    limit="auto"
    for source in sources:
        if not isinstance(source,dict):raise ValueError("Resource settings must be an object")
        block=source.get("execution_limits",source)
        if not isinstance(block,dict):raise ValueError("execution_limits must be an object")
        if "max_rss_bytes" in block:
            raw=block["max_rss_bytes"]
            if raw=="auto":limit="auto";continue
            limit=float(raw)
        elif "max_process_tree_memory_gib" in block:
            raw=block["max_process_tree_memory_gib"];limit=float(raw)*1024**3
        else:continue
        if isinstance(raw,bool) or not math.isfinite(limit) or limit<1:
            raise ValueError("Explicit process memory limit must be positive, finite and at least one byte")
    if limit=="auto":return limit
    if not math.isfinite(limit) or limit<=0:raise ValueError("Explicit process memory limit must be positive and finite")
    return int(limit)


def run_physics_attempts(*,workspace,project,folder,scene,test,chrono_python,budget,
                         timeout_seconds,numerical_settings=None,modeling_policy=None,runner=run_bounded_subprocess,max_attempts=4,history_binding=None):
    """Resource waits are outside active work; every retry gets a fresh deadline."""
    root=Path(folder)
    execution_started=False
    def failure(result):return {**result,"execution_started":execution_started,**retained_prefixes(root)}
    try:
        policy=load_declared_resource_requirement(project,modeling_policy)
        rss_limit=_rss_limit(project,modeling_policy)
    except (OSError,ValueError,TypeError) as exc:
        return failure(resource_policy_required_result(reason=str(exc)))
    if max_attempts < 1:raise ValueError("At least one physical attempt is required")
    for retry in range(max_attempts):
        try:budget.limit(timeout_seconds)
        except ActiveBudgetExhausted as exc:
            return failure({"status":"ACTIVE_BUDGET_EXHAUSTED","issues":[str(exc)],"partial_evidence":str(root)})
        except ObservedWaitRecoveryRequired as exc:
            return failure(resource_policy_required_result(reason=str(exc)))
        target=root if retry==0 else root/f"retry-{retry}"
        target.mkdir(parents=True,exist_ok=retry==0)
        if policy:
            if not budget.can_observe_wait:
                return failure(resource_policy_required_result(reason="This worker cannot pause its parent deadline during a nested resource wait; resume this simulation outside that worker"))
            with budget.observed_wait("resource_admission"):
                admission=wait_for_observed_resources(SystemProbe(disk_path=project or workspace),
                    policy["required_available_memory_gib"],max_wait_seconds=policy["max_resource_wait_seconds"],
                    poll_seconds=policy["resource_poll_seconds"],should_stop=budget.stop.is_set)
            write_json_atomic(target/"resource-admission.json",admission)
            if admission["status"]!="ready":
                return failure({**resource_policy_required_result(reason="Declared resources did not become ready within the finite wait",evidence=admission),
                        "partial_evidence":str(target)})
        try:effective=budget.limit(timeout_seconds)
        except ActiveBudgetExhausted as exc:
            return failure({"status":"ACTIVE_BUDGET_EXHAUSTED","issues":[str(exc)],"partial_evidence":str(target)})
        except ObservedWaitRecoveryRequired as exc:
            return failure(resource_policy_required_result(reason=str(exc)))
        project_limited=effective<timeout_seconds
        trace=target/"physical.trace.json"
        request=target/"request.json"
        limits={"timeout_seconds":effective,"max_rss_bytes":rss_limit}
        try:
            with budget.active("physical_simulation"):
                # Starting the scoped observation/checkpoint also costs time.
                # Recompute after that setup, immediately before the launch.
                effective=budget.limit(timeout_seconds)
                project_limited=effective<timeout_seconds
                limits["timeout_seconds"]=effective
                write_json_atomic(request,{"scene":scene,"test":test,"output":str(trace),
                                           "execution_limits":limits,"numerical_settings":numerical_settings or {},"history_binding":history_binding or {}})
                from ..trajectory.configuration import save_simulation_configuration
                save_simulation_configuration(target,interpreter=chrono_python,test=test,limits=limits,numerical=numerical_settings,history_binding=history_binding)
                from .execution import geometry_inputs,run_compute
                result=run_compute([str(chrono_python),"-X","utf8","-m",
                    "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",str(request)],
                    workspace=workspace,project=project,inputs=[request,*geometry_inputs(scene)],outputs=[target],
                    cwd=workspace,runner=runner,**limits)
        except ActiveBudgetExhausted as exc:
            return failure({"status":"ACTIVE_BUDGET_EXHAUSTED","issues":[str(exc)],"partial_evidence":str(target)})
        except ObservedWaitRecoveryRequired as exc:
            return failure(resource_policy_required_result(reason=str(exc)))
        except BoundedSubprocessError as exc:
            execution_started=exc.code in {"BOUNDED_SUBPROCESS_TIMEOUT","BOUNDED_SUBPROCESS_RSS"}
            write_json_atomic(target/"execution-failure.json",{"code":exc.code,"message":exc.message,"execution_limits":limits,"memory_budget":getattr(exc,"memory_budget",None),**retained_prefixes(target)})
            (target/"stderr.log").write_text(exc.message,encoding="utf-8")
            if exc.code=="BOUNDED_SUBPROCESS_MEMORY_UNAVAILABLE":
                return failure({**resource_policy_required_result(reason=exc.message,evidence=getattr(exc,"memory_budget",None)),"partial_evidence":str(target)})
            if exc.code=="BOUNDED_SUBPROCESS_RSS":
                if policy is not None and retry+1<max_attempts:continue
                return failure({**resource_policy_required_result(reason="RSS limit exceeded; no declared recovery policy or finite retries exhausted",
                    evidence={"folder":str(target),"message":exc.message}),"partial_evidence":str(target)})
            if exc.code=="BOUNDED_SUBPROCESS_TIMEOUT":
                return failure({"status":"ACTIVE_BUDGET_EXHAUSTED" if project_limited else "SIMULATION_TIMEOUT",
                    "issues":[{"test_id":test["id"],"timeout_seconds":effective,"message":exc.message,
                               "partial_evidence":str(target),"log_path":str(target/"stderr.log")}],
                    "partial_evidence":str(target)})
            return failure({"status":"HARNESS_ERROR","issues":[str(exc)],"partial_evidence":str(target)})
        execution_started=True
        write_json_atomic(target/"memory-budget.json",getattr(result,"memory_budget",None) or {"requested_max_rss_bytes":rss_limit})
        (target/"stdout.log").write_bytes(result.stdout)
        (target/"stderr.log").write_bytes(result.stderr)
        if result.returncode:
            return failure({"status":"HARNESS_ERROR","issues":[result.stderr.decode("utf-8",errors="replace")],"partial_evidence":str(target)})
        return {"status":"ready","folder":str(target),"trace_file":str(trace),"retry_count":retry,
                **(retained_prefixes(root) if retry else {})}
