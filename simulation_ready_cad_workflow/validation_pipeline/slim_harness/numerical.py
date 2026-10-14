"""Bounded fixed-step retries from the earlier native experiment runner; never edit the experiment."""
from __future__ import annotations

from pathlib import Path
import math
import time

from ..mechanism.io import write_json_atomic
from .trace_storage import reuse_trace

RETRYABLE={"contact_penetration","solver_nonconvergence","nonfinite_solved_state"}
NUMERICAL_POLICY="fixed_1ms_max_two_halvings"


def validate_max_step_seconds(value):
    value=float(value)
    if not math.isfinite(value) or value<=0 or value>.001:
        raise ValueError("Physics max step must be positive, finite and at most 0.001 seconds")
    return value


def validate_solver_iterations(value):
    if isinstance(value,bool) or not isinstance(value,int) or value<=0:
        raise ValueError("Solver iterations must be a positive integer")
    return value


def effective_initial_step_seconds(test,max_step_s=.001):
    requested=validate_max_step_seconds(max_step_s)
    author=float(test.get("step_s",.001))
    if not math.isfinite(author) or author<=0:
        raise ValueError("Test step must be positive and finite")
    return min(.001,author,requested)


def numerical_policy_matches(trace,test,max_step_s=.001,*,solver_iterations=160):
    """Match the initial schedule, not merely the last retry's smaller step."""
    try:
        settings=trace["numerical_settings"]
        initial=effective_initial_step_seconds(test,settings["requested_max_step_s"])
        return (trace.get("numerical_policy")==NUMERICAL_POLICY
                and settings.get("solver_iterations",160)==validate_solver_iterations(solver_iterations)
                and settings["effective_initial_step_s"]==initial
                and trace["numerical_attempts"][0]["max_step_s"]==initial
                and initial==effective_initial_step_seconds(test,max_step_s))
    except (KeyError,IndexError,TypeError,ValueError):
        return False


def run_fixed_attempts(run_once,test,output,*,max_step_s=.001,solver_iterations=160):
    requested=validate_max_step_seconds(max_step_s)
    solver_iterations=validate_solver_iterations(solver_iterations)
    initial_step=effective_initial_step_seconds(test,requested)
    output=Path(output)
    started=time.monotonic()
    step=initial_step
    attempts=[]
    for index in range(3):
        path=output.with_suffix(".numerical")/f"a{index+1:03d}"/output.name
        path.parent.mkdir(parents=True,exist_ok=False)
        result=run_once(path,step)
        failure=(result.get("failure") or {}).get("kind")
        attempts.append({"trace_file":str(path),"max_step_s":step,
                         "actual_step_s":result.get("step_s"),"failure_kind":failure,
                         "physical_pass":result["physical_pass"],"elapsed_seconds":result.get("elapsed_seconds")})
        if failure not in RETRYABLE or index==2:
            break
        step/=2
    trace=reuse_trace(path,output)
    metadata=dict(trace.metadata if hasattr(trace,"metadata") else trace)
    metadata.update(numerical_attempts=attempts,numerical_policy=NUMERICAL_POLICY,
                    numerical_settings={"requested_max_step_s":requested,"effective_initial_step_s":initial_step,
                                        "solver_iterations":solver_iterations},
                    numerical_failure_unresolved=failure in RETRYABLE,
                    evaluation_wall_seconds=time.monotonic()-started)
    if failure in RETRYABLE:metadata["physical_pass"]=False
    write_json_atomic(output,metadata)
    return metadata
