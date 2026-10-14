"""Reuse identical completed development physics after formal preflight review.

No development verdict is promoted. Formal source-motion audit and independent
final acceptance still run. Artifact paths may move; physical inputs may not.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from ..mechanism.io import write_json_atomic
from .implementation import native_implementation_binding
from .trace_storage import file_digest, open_trace, release_trace_cache, reuse_trace
from .measurement_registry import validate_trace_measurements
from .numerical import numerical_policy_matches, validate_max_step_seconds, validate_solver_iterations


def input_key(prepared, test):
    value=copy.deepcopy(prepared)
    value["tests"]=[copy.deepcopy(test)]
    for body in value["bodies"]:
        body.pop("source_stat",None)
        for key in ("source_file","mesh_file","audit_mesh_file"):
            if key in body:body[key]={"sha256":file_digest(Path(body[key]))}
        for component in body.get("source_components",[]):
            component["mesh_file"]={"sha256":file_digest(Path(component["mesh_file"]))}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def store_trial(prepared,test,trace_path,chrono_python,*,output_dir=None):
    trace_path=Path(trace_path)
    request_path=trace_path.parent/"request.json"
    request=json.loads(request_path.read_text(encoding="utf-8"))
    if (request.get("scene")!=prepared or request.get("test")!=test
            or Path(request.get("output","")).resolve()!=trace_path.resolve()):
        raise ValueError("Completed physics must match its program-owned request")
    trace=open_trace(trace_path)
    try:
        if trace.get("test_id")!=test["id"] or not trace.get("physical_pass") or trace.get("failure") or trace.get("numerical_failure_unresolved"):
            return None
        if trace.get("execution_implementation")!=native_implementation_binding():return None
        if not numerical_policy_matches(trace,test,request.get("numerical_settings",{}).get("max_step_s",.001),
                solver_iterations=request.get("numerical_settings",{}).get("solver_iterations",160)):
            return None
        validate_trace_measurements(trace,prepared.get("registered_measurements",{}).get(test["id"]))
        record={"schema_version":"slim_development_physics_cache_v1","input_key":input_key(prepared,test),
                "prepared":prepared,"test":test,"trace_file":str(trace_path.resolve()),
                "request_file":str(request_path.resolve()),"request_sha256":file_digest(request_path),
                "trace_sha256":file_digest(trace_path),"chrono_python":str(Path(chrono_python).resolve())}
        target=(Path(output_dir) if output_dir is not None else trace_path.parent)/"physics-cache.json"
        write_json_atomic(target,record)
        return str(target)
    finally:release_trace_cache(trace)


def reuse_trial(workspace,prepared,test,trace_path,chrono_python,*,max_step_s=.001,solver_iterations=160):
    max_step_s=validate_max_step_seconds(max_step_s)
    solver_iterations=validate_solver_iterations(solver_iterations)
    started=time.monotonic()
    target_key=None
    candidates=sorted((Path(workspace)/".runtime_runs/author_tools").glob("simulate-*/tests/*/physics-cache.json"),
                      key=lambda p:p.stat().st_mtime_ns,reverse=True)
    for path in candidates:
        trace=None
        try:
            record=json.loads(path.read_text(encoding="utf-8"))
            if record.get("schema_version")!="slim_development_physics_cache_v1":continue
            if record["test"]!=test or record["chrono_python"]!=str(Path(chrono_python).resolve()):continue
            if target_key is None:target_key=input_key(prepared,test)
            if record["input_key"]!=target_key or input_key(record["prepared"],record["test"])!=target_key:continue
            source=Path(record["trace_file"])
            if file_digest(Path(record["request_file"]))!=record["request_sha256"]:continue
            if file_digest(source)!=record["trace_sha256"]:continue
            trace=open_trace(source)
            if (trace.get("execution_implementation")!=native_implementation_binding()
                    or trace.get("test_id")!=test["id"] or not trace.get("physical_pass") or trace.get("failure")
                    or trace.get("numerical_failure_unresolved")):continue
            if not numerical_policy_matches(trace,test,max_step_s,solver_iterations=solver_iterations):continue
            validate_trace_measurements(trace,prepared.get("registered_measurements",{}).get(test["id"]))
            reused=reuse_trace(source,trace_path)
            metadata=reused.metadata if hasattr(reused,"metadata") else reused
            metadata["physics_cache_reuse"]={"receipt":str(path),"source_trace":str(source),
                "original_physics_elapsed_seconds":trace.get("elapsed_seconds"),"reuse_elapsed_seconds":time.monotonic()-started}
            write_json_atomic(trace_path,metadata)
            return reused
        except (OSError,ValueError,KeyError,TypeError):
            # Missing/changed records are a cache miss, never an acceptance.
            continue
        finally:
            if trace is not None:release_trace_cache(trace)
    return None
