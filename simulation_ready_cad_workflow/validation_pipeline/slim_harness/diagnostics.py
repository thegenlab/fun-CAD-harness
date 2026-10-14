"""Record force overlays for an accepted scene without rewriting its result."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import run_bounded_subprocess
from .audit import audit_motion
from .evidence import export_replay
from .publication import read_result
from .trace_storage import open_trace, release_trace_cache
from .workflow import CHRONO_PYTHON, relativize_preview_paths


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_recorded_poses(left, right):
    a,b=left.get("samples",[]),right.get("samples",[])
    return bool(a) and len(a)==len(b) and all(
        x["time_s"]==y["time_s"] and x["body_poses"]==y["body_poses"] for x,y in zip(a,b))


def record_diagnostics(workspace: Path, run: Path, *, regions_only=False):
    started=time.monotonic()
    workspace,run=Path(workspace).resolve(),Path(run).resolve()
    outcome=read_result(run)
    if not outcome.get("eligible_pass"):
        raise ValueError("Force diagnostics require the accepted, source-bound scene")
    geometry=read(Path(outcome["attempt"])/"geometry/geometry.json")
    folder=run/"diagnostics"/(("regions_" if regions_only else "forces_")+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    folder.mkdir(parents=True)
    preview_file=run/"slim_evidence/preview.json"
    preview=read(preview_file)
    write_json_atomic(folder/"preview.before.json",preview)
    inputs={path:digest(path) for path in outcome["source_bindings"]}
    summaries=[]
    for baseline in outcome["tests"]:
        test_id=baseline["test_id"]
        original_trace=Path(baseline["trace_file"])
        request=read(original_trace.parent/"request.json")
        old=open_trace(original_trace)
        output=folder/test_id;output.mkdir()
        trace_path=output/"physical.trace.json"
        if regions_only:
            trace={**old,"connections":request["scene"].get("joints",[])}
            audit=baseline["geometry_audit"]
        else:
            write_json_atomic(output/"request.json",{**request,"output":str(trace_path)})
            child=run_bounded_subprocess([CHRONO_PYTHON,"-X","utf8","-m",
                "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",str(output/"request.json")],
                cwd=workspace,timeout_seconds=900,max_rss_bytes="auto")
            (output/"stdout.log").write_bytes(child.stdout);(output/"stderr.log").write_bytes(child.stderr)
            if child.returncode!=0:raise RuntimeError(child.stderr.decode("utf-8",errors="replace"))
            trace=open_trace(trace_path)
            if not trace.get("physical_pass"):raise RuntimeError(f"Diagnostic physics failed: {trace_path}")
            if not any(row.get("force_evidence",{}).get("available") for row in trace["samples"]):
                raise RuntimeError(f"Force telemetry was not recorded: {trace_path}")
            if same_recorded_poses(old,trace):
                audit={**baseline["geometry_audit"],"reused_from":str(original_trace.parent/"geometry-audit.json"),
                    "reuse_reason":"All original-source body poses and solver times match exactly at every step"}
                write_json_atomic(output/"geometry-audit.json",audit)
            else:
                audit=audit_motion(request["scene"],trace["samples"],output/"geometry-audit.json")
            if not audit["passed"]:raise RuntimeError(f"Diagnostic original-geometry audit failed: {output}")
        assets=export_replay(geometry,trace,output/"replay")
        release_trace_cache(old)
        release_trace_cache(trace)
        source_trace=original_trace if regions_only else trace_path
        motion={"id":test_id,"load_case_id":test_id,"label":test_id.replace("_"," "),
            "motion_model_path":assets["model_file"],"scene_path":assets["scene_file"],"replay_path":assets["replay_file"],
            "motion_trace":str(source_trace),"motion_frame_count":min(720,len(trace["samples"])),
            "replay_version":digest(assets["replay_file"]),"diagnostic":True,
            "connection_count":len(trace.get("connections",[])),"force_evidence_available":not regions_only}
        if any(digest(path)!=expected for path,expected in inputs.items()):
            raise RuntimeError("Source changed during diagnostics; original results and prior preview retained")
        motions=preview.get("motions",[])
        preview["motions"]=[motion if row["id"]==test_id else row for row in motions]
        if not any(row["id"]==test_id for row in motions):preview["motions"].append(motion)
        primary=preview["motions"][0]
        preview.update({key:value for key,value in primary.items() if key not in {"id","label","load_case_id"}})
        preview["diagnostics"]={"kind":"regions" if regions_only else "forces","completed":len(summaries)+1,
            "total":len(outcome["tests"]),"directory":str(folder),"elapsed_seconds":time.monotonic()-started}
        relativize_preview_paths(workspace,preview)
        write_json_atomic(preview_file,preview)
        summaries.append({"test_id":test_id,"trace_file":str(source_trace),"physical_pass":trace["physical_pass"],
            "geometry_audit_pass":audit["passed"],"geometry_audit_reused":bool(audit.get("reused_from")),"replay":assets})
        write_json_atomic(folder/"diagnostics.json",{"source_bindings":inputs,"tests":summaries,"elapsed_seconds":time.monotonic()-started,
            "scope":"Recorded rigid-body force diagnostics; existing acceptance result unchanged"})
        print(json.dumps({"run":run.name,"project":run.parent.parent.name,"completed_test":test_id,"regions_only":regions_only}),flush=True)
    return {"directory":str(folder),"tests":len(summaries),"elapsed_seconds":time.monotonic()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path);parser.add_argument("--workspace",type=Path,default=Path.cwd())
    parser.add_argument("--regions-only",action="store_true")
    args=parser.parse_args()
    print(json.dumps(record_diagnostics(args.workspace,args.run,regions_only=args.regions_only)))


if __name__=="__main__":main()
