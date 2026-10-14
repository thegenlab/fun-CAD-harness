"""Bounded advisory review of a changed simulation task; no acceptance gate."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import time

from ..cold_start.worker import CodexCliWorker
from ..mechanism.io import write_json_atomic
from .budget import ActiveBudgetExhausted,BudgetedWorker


REVIEW_SOFT_SECONDS={"light":60.,"medium":120.,"high":180.}
REVIEW_GRACE_SECONDS=60.
TIME_REMINDER="审核时间已到。请立即基于已查看的材料给出结论；尚未完成的部分明确列为不确定。不要重新开始或继续扩大审核。"
CHANGE_REVIEW_SCHEMA={"type":"object","additionalProperties":False,
    "required":["conclusion","issues","evidence_paths"],"properties":{
        "conclusion":{"type":"string"},
        "issues":{"type":"array","items":{"type":"string"}},
        "evidence_paths":{"type":"array","items":{"type":"string"}}}}


def _outputs(directory):
    files={};texts={}
    for name in ("events.jsonl","stderr.log","response.json","response.partial.json"):
        path=directory/name
        if path.is_file():
            files[name]=str(path)
            texts[name]=path.read_text(encoding="utf-8",errors="replace")
    messages=[]
    for line in texts.get("events.jsonl","").splitlines():
        try:event=json.loads(line)
        except ValueError:continue
        if not isinstance(event,dict):continue
        item=event.get("item")
        if isinstance(item,dict) and item.get("type")=="agent_message" and isinstance(item.get("text"),str):
            if item["text"] not in messages:messages.append(item["text"])
    response=texts.get("response.partial.json") or texts.get("response.json") or ""
    raw="\n".join([*messages,response] if messages else [texts.get("events.jsonl",""),response])
    if texts.get("stderr.log"):raw+="\nSTDERR:\n"+texts["stderr.log"]
    return files,raw.strip()


def run_change_review(reviewer,workspace,prompt,outdir,*,tier_name=None,budget=None):
    """Return all advisory output, including partial text, without retry/approval.

    Codex CLI cannot steer this running readonly call in place. At the soft
    boundary it is stopped and the same retained independent session receives
    the one fixed reminder. Missing session identity never starts another
    review. The project budget can terminate either stage earlier.
    """
    if isinstance(reviewer,BudgetedWorker):
        budget=budget if budget is not None else reviewer.budget
        reviewer=reviewer.worker
    tier=tier_name or "light"
    tracker=getattr(budget,"tracker",None) if budget is not None else None
    workspace,outdir=Path(workspace).resolve(),Path(outdir).resolve()
    outdir.mkdir(parents=True,exist_ok=True)
    if any((outdir/name).exists() for name in ("initial","reminder","review-result.json")):
        raise ValueError("Use a fresh review output directory; previous output is preserved")
    unavailable=("Timed same-session review requires CodexCliWorker" if not isinstance(reviewer,CodexCliWorker) else
                 "Unknown review tier" if tier not in REVIEW_SOFT_SECONDS else
                 "Review timing requires the owning project budget tracker" if budget is not None and getattr(budget,"tier",None) is not None and tracker is None else None)
    if unavailable:
        result={"schema_version":"slim_task_change_review_v1","status":"partial","decision_owner":"author",
                "review_is_acceptance_gate":False,"tier":tier,"stop_reason":"review_execution_unavailable",
                "error":unavailable,"raw_output":unavailable,"phases":[],"reminder_count":0,
                "finished_at":datetime.now(timezone.utc).isoformat()}
        write_json_atomic(outdir/"review-result.json",result)
        return result
    remaining=lambda:float(tracker.remaining_active) if tracker is not None else math.inf
    soft=REVIEW_SOFT_SECONDS[tier];started=time.monotonic()
    soft_deadline=started+soft;hard_deadline=soft_deadline+REVIEW_GRACE_SECONDS
    report={"schema_version":"slim_task_change_review_v1","status":"partial","decision_owner":"author",
        "review_is_acceptance_gate":False,"tier":tier,"started_at":datetime.now(timezone.utc).isoformat(),
        "soft_seconds":soft,"hard_after_soft_seconds":REVIEW_GRACE_SECONDS,
        "project_remaining_at_start":remaining() if tracker is not None else None,
        "reminder_count":0,"reminder_method":"interrupt_then_resume_same_independent_session",
        "in_place_steer":False,"phases":[],"raw_output":""}
    raw_parts=[]
    policy=f"审核agent每次最多审核 {soft/60:g} min，到时间强行出结论。审核仅提供意见，由作者决定是否接受仿真任务变更。"
    previous_controller=getattr(reviewer,"budget_controller",None)
    reviewer.budget_controller=None  # Review caps are wall time, not simulation limits.
    active_started=None

    def phase(name,text,limit,session_id=None):
        directory=outdir/name;phase_started=time.monotonic()
        entry={"phase":name,"directory":str(directory),"timeout_seconds":limit,
               "started_at":datetime.now(timezone.utc).isoformat(),"requested_session_id":session_id}
        report["phases"].append(entry)
        try:
            result=reviewer.run_turn(workspace,text,directory,limit,readonly=True,fresh=session_id is None,
                response_schema=CHANGE_REVIEW_SCHEMA,retain_independent_session=True,independent_session_id=session_id)
            entry.update(raw_result=result,process_pid=result.get("process_pid"),exit_code=result.get("exit_code"),
                         observed_session_id=result.get("observed_session_id"),
                         termination=result.get("provider_error"))
            return result
        finally:
            entry["elapsed_seconds"]=time.monotonic()-phase_started
            entry["finished_at"]=datetime.now(timezone.utc).isoformat()
            files,raw=_outputs(directory);entry["output_files"]=files
            if raw:raw_parts.append(f"[{name}]\n{raw}")

    try:
        with budget.active("worker_elapsed_upper_estimate") if budget is not None else nullcontext():
            active_started=time.monotonic()
            try:
                limit=min(max(0.,soft_deadline-time.monotonic()),remaining())
                if limit<=0:raise ActiveBudgetExhausted("No project budget remains for review")
                first=phase("initial",policy+"\n\n"+prompt,limit)
                if first.get("status")!="provider_error":
                    report.update(status="completed",stop_reason="review_returned")
                elif remaining()<=0:
                    report.update(status="partial",stop_reason="project_budget_exhausted")
                elif first.get("provider_error",{}).get("kind")!="timeout":
                    report["stop_reason"]="review_failed_without_retry"
                elif time.monotonic()<soft_deadline:
                    report["stop_reason"]="review_stopped_before_soft_deadline"
                elif first.get("provider_error",{}).get("cleanup_confirmed") is not True:
                    report["stop_reason"]="initial_cleanup_unconfirmed_no_resume"
                elif not first.get("observed_session_id"):
                    report["stop_reason"]="independent_session_unavailable_no_restart"
                else:
                    extra=min(REVIEW_GRACE_SECONDS,max(0.,hard_deadline-time.monotonic()),remaining())
                    if extra<=0:
                        report["stop_reason"]="hard_deadline" if remaining()>0 else "project_budget_exhausted"
                    else:
                        sid=first["observed_session_id"]
                        report.update(reminder_count=1,reminder_text=TIME_REMINDER,review_session_id=sid)
                        write_json_atomic(outdir/"reminder-boundary.json",{
                            "method":report["reminder_method"],"in_place_steer":False,"reminder_count":1,
                            "session_id":sid,"reminder_text":TIME_REMINDER,"remaining_phase_seconds":extra,
                            "initial_process_pid":first.get("process_pid"),"initial_termination":first.get("provider_error"),
                            "at":datetime.now(timezone.utc).isoformat()})
                        second=phase("reminder",TIME_REMINDER,extra,sid)
                        if second.get("observed_session_id") not in (None,sid):
                            report["stop_reason"]="unexpected_session_identity"
                        elif second.get("status")!="provider_error":
                            report.update(status="completed",stop_reason="review_returned_after_reminder")
                        else:
                            report["stop_reason"]="project_budget_exhausted" if remaining()<=0 else "hard_deadline" if second.get("provider_error",{}).get("kind")=="timeout" else "reminder_failed_without_retry"
            finally:
                if budget is not None:budget.record_unclassified_provider(time.monotonic()-active_started)
    except ActiveBudgetExhausted as exc:
        report.update(status="partial",stop_reason="project_budget_exhausted",error=str(exc))
    except Exception as exc:
        report.update(status="partial",stop_reason="review_execution_error",error=f"{type(exc).__name__}: {exc}")
    finally:
        reviewer.budget_controller=previous_controller
        report.update(raw_output="\n\n".join(raw_parts),wall_seconds=time.monotonic()-started,
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      project_remaining_at_end=remaining() if tracker is not None else None)
        write_json_atomic(outdir/"review-result.json",report)
    return report
