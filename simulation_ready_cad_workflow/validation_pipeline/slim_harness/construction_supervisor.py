"""A project supervisor enforces effective time even inside native geometry code.

Only the child process tree created here may be stopped. The parent then emits
a completion summary from durable evidence, including interrupted partial work.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import threading
import time
import uuid

from ..runtime.system_probe import terminate_process_tree
from ..mechanism.io import write_json_atomic
from ..benchmark_sandbox.formal_limits import FORMAL_WORKFLOW_STOP_SECONDS
from .budget import PROVIDER_BACKOFF_CAP_SECONDS, PROVIDER_RECOVERY_RETRIES, provider_outage, recovery_delay
from .construction import budget_for,summary
from .construction_budget import BudgetExhausted


def _owned_stop_reason(project):
    raw = os.environ.get('CADHARNESS_SANDBOX_STOP_FILE')
    if not raw:
        return None
    path = Path(raw)
    if (not path.is_absolute() or path.resolve().parent != Path(project).resolve()
            or not path.name.startswith('sandbox-stop-') or not path.is_file()):
        return None
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        record = {}
    return 'budget_exhausted' if isinstance(record, dict) and record.get('reason') == 'budget_exhausted' else 'operator_stop'


def _stopped_result(project, reason, previous=None):
    state = json.loads((Path(project) / 'workflow.json').read_text(encoding='utf-8'))
    state.update(phase='budget_exhausted' if reason == 'budget_exhausted' else 'stopped',
                 stop_reason=reason, complete_instance_pass=False, active_turn={})
    write_json_atomic(Path(project) / 'workflow.json', state)
    result = {'status': 'BUDGET_EXHAUSTED' if reason == 'budget_exhausted' else 'INTERRUPTED',
              'stop_reason': reason, 'complete_instance_pass': False, 'issues': []}
    if previous:
        result['last_result'] = previous
    return result


@contextmanager
def _supervisor_stop_signals():
    requested = {'signal': None}
    previous = {}
    if threading.current_thread() is threading.main_thread():
        def request(number, _frame):
            requested['signal'] = number
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, request)
    try:
        yield requested
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _process_started_at(pid):
    import psutil
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _active_sandbox(stop_file, workflow_pid, workflow_started_at):
    if workflow_started_at is None:
        return None
    try:
        record=json.loads(Path(str(stop_file)+'.active.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return None
    if (not isinstance(record,dict) or record.get('status')!='running'
            or record.get('stop_file')!=str(stop_file)
            or type(record.get('workflow_pid')) is not int or record['workflow_pid']!=workflow_pid
            or type(record.get('workflow_started_at_epoch')) not in (int,float)
            or not math.isfinite(record['workflow_started_at_epoch'])
            or abs(record['workflow_started_at_epoch']-workflow_started_at)>.001
            or type(record.get('controller_pid')) is not int or record['controller_pid']<=0):
        return None
    return record


def _wait_for_provider(project, budget, attempt):
    """Back off outside the work budget; an owned stop ends the wait early."""
    from ..cold_start.runtime_status import RuntimeHeartbeat
    delay = recovery_delay(attempt, PROVIDER_BACKOFF_CAP_SECONDS)
    path = Path(project) / 'workflow.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    phase = state.get('phase')
    state.update(phase='recovery_wait', stop_reason='', recovery_attempt=attempt + 1, recovery_delay_seconds=delay,
                 recovery_from_phase=phase, updated_at=datetime.now(timezone.utc).isoformat())
    write_json_atomic(path, state)
    deadline = time.monotonic() + delay
    # The heartbeat shows the wait and keeps this runner owning the project.
    with RuntimeHeartbeat(project, state), budget.activity('provider_wait'):
        while not _owned_stop_reason(project) and deadline > time.monotonic():
            time.sleep(min(1., deadline - time.monotonic()))
        state.pop('recovery_from_phase')
        state.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat())
        write_json_atomic(path, state)


def execute_project(workspace,project,*,executable='codex',one_revision=False,feedback_mode="full"):
    from .cli import resume_project
    budget=budget_for(project);result={};reason='';outages=0
    try:
        with budget.activity('workflow'):
            while True:
                reason = _owned_stop_reason(project)
                if reason:
                    result = _stopped_result(project, reason, result)
                    break
                budget.check_available()
                if one_revision:
                    # Preserve the diagnostic CLI's single-revision behavior,
                    # but still account for all work and always summarize it.
                    from .workflow import SlimWorkflow
                    result=SlimWorkflow(workspace,project,executable=executable,feedback_mode=feedback_mode).run(one_revision=True)
                else:
                    result=resume_project(workspace,project,executable=executable,in_process=True,feedback_mode=feedback_mode)
                reason = _owned_stop_reason(project)
                if reason:
                    result = _stopped_result(project, reason, result)
                    break
                if one_revision:break
                # A provider outage resumes the same session after a bounded backoff.
                if result.get('status')=='provider_error' and provider_outage(result.get('provider_error')):
                    if outages>=PROVIDER_RECOVERY_RETRIES:break
                    _wait_for_provider(project,budget,outages);outages+=1
                    continue
                outages=0
                provider=result.get('provider_error',{}).get('kind')
                retryable=result.get('status') in {
                    'needs_revision','NEEDS_REVISION','unsupported','harness_defect','HARNESS_ERROR'} or (
                    result.get('status')=='provider_error' and provider in {'timeout','response'})
                if not retryable:break
                # Preserve the same author session and evidence; allow a new
                # strategy while time remains, instead of requiring a human
                # resume for every incomplete author turn.
                state=json.loads((Path(project)/'workflow.json').read_text())
                from .feedback_ablation import transform_feedback
                state['last_construction_feedback']=transform_feedback(result,feedback_mode)
                write_json_atomic(Path(project)/'workflow.json',state)
    except BudgetExhausted as exc:
        reason='budget_exhausted_or_summary_reserve';result={'status':'BUDGET_EXHAUSTED','issues':[str(exc)]}
    except KeyboardInterrupt:
        reason='operator_stop';result=_stopped_result(project,reason,result)
    except Exception as exc:
        import traceback
        reason=type(exc).__name__;result={'status':'HARNESS_ERROR','issues':[str(exc)],'traceback':traceback.format_exc()}
    finally:
        write_json_atomic(Path(project)/'construction_result.json',result)
        summary(project,reason=reason or result.get('status','stopped'),result=result)
    return result


def supervise_project(workspace,project,*,executable='codex',one_revision=False,feedback_mode="full"):
    from .workflow import assert_inactive
    project=Path(project).resolve();workspace=Path(workspace).resolve();assert_inactive(project)
    budget=budget_for(project)
    if budget.snapshot()['work_remaining_seconds']<=0:return summary(project,reason='budget_exhausted')
    # A resumed process must never reuse a previous turn's result if it is
    # interrupted before writing its own. Historical summaries stay on disk.
    old_result=project/'construction_result.json'
    if old_result.exists():old_result.rename(project/f'construction_result.previous-{time.time_ns()}.json')
    env=os.environ.copy();env['SIMREADY_CONSTRUCTION_PROJECT']=str(project)
    code_root=Path(__file__).resolve().parents[3]
    env['PYTHONPATH']=str(code_root)+os.pathsep+env.get('PYTHONPATH','')
    stop_file=project/('sandbox-stop-'+uuid.uuid4().hex+'.json')
    env['CADHARNESS_SANDBOX_STOP_FILE']=str(stop_file)
    command=[sys.executable,'-m','simulation_ready_cad_workflow.validation_pipeline.slim_harness',
        '--workspace',str(workspace),'--codex',executable,'_construction-execute','--project',project.name]
    if one_revision:command.append('--one-revision')
    if feedback_mode and feedback_mode!="full":command.extend(['--feedback-mode',feedback_mode])
    log=project/f'construction-supervisor-{time.time_ns()}.log'
    started=time.monotonic();stop_reason=None;shutdown_errors=[]
    process=None;workflow_started_at=None;decision_elapsed=None;cleanup_wall_seconds=0.
    cooperative={'requested':False,'wait_seconds':90,'owned_workflow_exit_confirmed':False,'fallback_used':False}
    # The external process keeps the clock alive even if native CAD holds
    # the child Python GIL longer than its internal heartbeat lease.
    with _supervisor_stop_signals() as signal_stop, log.open('w',encoding='utf-8') as stream:
        try:
            with budget.activity('workflow_supervisor') as active:
                process=subprocess.Popen(command,cwd=code_root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
                workflow_started_at=_process_started_at(process.pid)
                write_json_atomic(project/'construction_supervisor.json',{'pid':process.pid,'status':'running','log':str(log),
                    'workflow_started_at_epoch':workflow_started_at,'sandbox_stop_file':str(stop_file)})
                while process.poll() is None:
                    if signal_stop['signal'] is not None or stop_file.is_file():
                        stop_reason='operator_stop';decision_elapsed=time.monotonic()-started;break
                    if budget.snapshot()['work_remaining_seconds']<=0:
                        stop_reason='budget_exhausted';decision_elapsed=time.monotonic()-started;break
                    try:active.check_available()
                    except BudgetExhausted:
                        stop_reason='budget_exhausted';decision_elapsed=time.monotonic()-started;break
                    time.sleep(.5)
        except KeyboardInterrupt:
            stop_reason='operator_stop';decision_elapsed=time.monotonic()-started
        finally:
            cleanup_started=time.monotonic()
            # The supervisor's activity is closed before any cleanup wait.
            # Only a matching active sandbox receives a graceful export window.
            if process is not None and process.poll() is None:
                active_sandbox=_active_sandbox(stop_file,process.pid,workflow_started_at)
                if active_sandbox is not None:
                    try:
                        write_json_atomic(stop_file,{'schema_version':1,
                            'reason':stop_reason or 'supervisor_stopped',
                            'decision_elapsed_wall_seconds':decision_elapsed,
                            'workflow_pid':process.pid,'workflow_started_at_epoch':workflow_started_at,
                            'controller_pid':active_sandbox['controller_pid']})
                        cooperative.update(requested=True,controller_pid=active_sandbox['controller_pid'],
                                           stop_file=str(stop_file),wait_seconds=FORMAL_WORKFLOW_STOP_SECONDS)
                        process.wait(timeout=FORMAL_WORKFLOW_STOP_SECONDS)
                        cooperative['owned_workflow_exit_confirmed']=process.poll() is not None
                    except subprocess.TimeoutExpired:
                        cooperative['wait_expired']=True
                        shutdown_errors.append('Sandbox cleanup was not confirmed within the cooperative stop window; owned-tree fallback required.')
                    except OSError as exc:
                        shutdown_errors.append('Cooperative sandbox stop: '+str(exc))
            if process is not None and process.poll() is None:
                cooperative['fallback_used']=True
                try:terminate_process_tree(process.pid,2.)
                except Exception as exc:shutdown_errors.append(f'Tree termination: {type(exc).__name__}: {exc}')
                if process.poll() is None:
                    try:process.kill()
                    except ProcessLookupError:pass
                    except OSError as exc:shutdown_errors.append(f'Child termination: {exc}')
            try:
                if process is not None:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Persist this uncertainty in the result, while still emitting
                # the required partial summary from already durable evidence.
                shutdown_errors.append('Owned child did not confirm exit within the bounded shutdown wait.')
            cleanup_wall_seconds=time.monotonic()-cleanup_started
    record={'status':stop_reason or 'finished','exit_code':process.returncode if process is not None else None,
        'wall_seconds':time.monotonic()-started,'log':str(log),'shutdown_errors':shutdown_errors,
        'sandbox_stop_file':str(stop_file),'cooperative_sandbox_stop':cooperative,
        'budget_decision_elapsed_wall_seconds':decision_elapsed if stop_reason=='budget_exhausted' else None,
        'stop_decision_elapsed_wall_seconds':decision_elapsed,'stop_signal':signal_stop['signal'],
        'cleanup_wall_seconds':cleanup_wall_seconds,
        'cleanup_accounting':'Supervisor activity ends before cleanup; other owned workflow activities retain the existing union-ledger charging until they stop'}
    write_json_atomic(project/'construction_supervisor.json',record)
    if stop_reason:
        state=json.loads((project/'workflow.json').read_text())
        phase='budget_exhausted' if stop_reason=='budget_exhausted' else 'stopped'
        state.update(phase=phase,stop_reason=stop_reason,complete_instance_pass=False,active_turn={})
        write_json_atomic(project/'workflow.json',state)
        heartbeat=project/'workflow_runtime.json'
        if heartbeat.exists():
            state=json.loads(heartbeat.read_text());state.update(status='stopped',phase=phase,stop_reason=stop_reason)
            write_json_atomic(heartbeat,state)
    result_path=project/'construction_result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else {'status':'INTERRUPTED','issues':['Inspect supervisor log']}
    if stop_reason=='operator_stop':
        result=_stopped_result(project,stop_reason,result)
        write_json_atomic(result_path,result)
    if shutdown_errors:result={**result,'issues':[*result.get('issues',[]),*shutdown_errors]}
    if stop_reason!='operator_stop':
        # Leave a motion to watch even when the run ended without a playable simulation.
        try:
            from .motion_preview import ensure_final_motion
            from .workflow import CHRONO_PYTHON
            request=json.loads((project/'workflow.json').read_text()).get('request') or {}
            ensure_final_motion(workspace,project,chrono_python=request.get('chrono_python') or CHRONO_PYTHON)
        except (Exception,KeyboardInterrupt) as exc:
            write_json_atomic(project/'final-motion.json',{'schema_version':'slim_final_motion_v1','status':'error',
                                                           'error':f'{type(exc).__name__}: {exc}'})
    completed=summary(project,reason=stop_reason or result.get('status','interrupted'),result=result)
    return {**result,'completion_summary':str(project/'completion_summary.json'),'completion':completed}
