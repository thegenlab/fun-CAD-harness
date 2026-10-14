"""Short, source-preserving cost measurements; never validation or cache input."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import time
from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import run_bounded_subprocess, BoundedSubprocessError
from .trace_storage import open_trace, open_trace_prefix, release_trace_cache


def measure_cost(prepared, test, folder, *, python, workspace, remaining_seconds, simulation_timeout,max_rss_bytes="auto"):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    probe=copy.deepcopy(test)
    registered=test["id"] in prepared.get("registered_measurements",{})
    # Exact registered measurement plans cannot be shortened. Measure a bounded
    # prefix of the original request instead, keeping its binding unchanged.
    if not registered:probe["duration_s"]=min(float(test["duration_s"]),.1)
    request=folder/"request.json";trace_path=folder/"physical.trace.json"
    cap=min(15.,float(simulation_timeout),float(remaining_seconds))
    write_json_atomic(request,{"scene":prepared,"test":probe,"output":str(trace_path),
                              "purpose":"cost_probe_only","execution_limits":{"timeout_seconds":cap}})
    command=[str(python),"-X","utf8","-m","simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",str(request)]
    env=os.environ.copy();env["SIMREADY_CONSTRUCTION_GUARDED_REQUEST"]=str(request.resolve())
    started=time.monotonic();failure=None;memory_budget=None
    try:
        execution=run_bounded_subprocess(command,cwd=workspace,env=env,timeout_seconds=cap,max_rss_bytes=max_rss_bytes)
        memory_budget=getattr(execution,"memory_budget",None)
        (folder/"stdout.log").write_bytes(execution.stdout);(folder/"stderr.log").write_bytes(execution.stderr)
        if execution.returncode:failure="runtime_error"
    except BoundedSubprocessError as exc:failure=exc.code;memory_budget=getattr(exc,"memory_budget",None)
    elapsed=time.monotonic()-started;observed=0.;trace=None;observed_file=None
    try:
        if trace_path.is_file():
            trace=open_trace(trace_path);observed_file=trace_path
        else:
            prefixes=sorted(folder.rglob("prefix.json"),key=lambda path:path.stat().st_mtime_ns)
            if prefixes:observed_file=prefixes[-1];trace=open_trace_prefix(observed_file)
        if trace is not None:
            samples=trace.get("samples",[])
            if len(samples):observed=float(samples[-1]["time_s"])
            if trace.get("numerical_failure_unresolved"):failure="numerical_failure"
    except (OSError,ValueError,KeyError,TypeError) as exc:
        failure=f"trace_error: {exc}"
    finally:
        if trace is not None:release_trace_cache(trace)
    estimate=elapsed*float(test["duration_s"])/observed if observed>0 else None
    result={"test_id":test["id"],"certified":False,"counts_as_test_execution":False,
            "scope":"diagnostic prefix only; no assertions accepted and no physics cache entry",
            "requested_full_duration_s":test["duration_s"],"prefix_duration_s":probe["duration_s"],"registered_test_unchanged":registered,
            "observed_simulated_seconds":observed,"elapsed_seconds":elapsed,"wall_limit_seconds":cap,
            "estimated_full_wall_seconds":estimate,"simulation_wall_limit_seconds":simulation_timeout,
            "estimate_basis":"prefix extrapolation includes startup and may miss later contacts; not a guaranteed bound",
            "failure":failure,"memory_budget":memory_budget,"request_file":str(request),"trace_file":str(observed_file) if observed_file else None,
            "required_action":"inspect_cost_and_choose_supported_revision" if failure or estimate is None or estimate>simulation_timeout else "run_complete_declared_test"}
    write_json_atomic(folder/"cost-probe.json",result)
    return result
