"""Bound construction source-motion audits without changing physical evidence.

The native simulation and this original-geometry audit have SEPARATE resource
caps: native solving defaults to 180 wall seconds, and original-geometry audit
to 600 seconds. The audit is clipped to remaining project work time; native
solving is not — it starts only while work time remains, then runs to its own
extension ceiling. 180 seconds is not a combined simulation-plus-audit
deadline.
The child runs unchanged audit_motion over the complete recorded trace. Timeout
means missing evidence, never a sampled/relaxed audit or a mechanical verdict.
Frozen evaluation continues to call audit_motion directly in workflow.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import BoundedSubprocessError, run_bounded_subprocess
from .construction_limits import ConstructionLimitError, ConstructionLimits


def run_motion_audit_bounded(prepared_scene_path, trace_path, output_path, *,
                             policy, remaining_seconds, cwd=None, cad_python=None,
                             test_id=None, max_rss_bytes="auto"):
    """Return the unchanged auditor result or actionable incomplete evidence.

    Only the child created by run_bounded_subprocess is subject to termination.
    Input scene/trace files are read-only. A unique child output prevents a
    previous PASS at output_path from being mistaken for this attempt's result.
    """
    prepared_scene_path,trace_path,output_path=map(lambda p:Path(p).resolve(),
                                                  (prepared_scene_path,trace_path,output_path))
    if output_path in {prepared_scene_path,trace_path}:
        raise ValueError('Audit output must differ from its scene and physical trace')
    limits=ConstructionLimits.from_policy(policy)
    if not math.isfinite(remaining_seconds) or remaining_seconds<=0:
        raise ConstructionLimitError('PROJECT_BUDGET_EXHAUSTED','No work time remains for a source-motion audit; write the completion summary.',
                                     trace_file=str(trace_path),test_id=test_id)
    timeout=min(limits.audit_timeout_seconds,float(remaining_seconds))
    output_path.parent.mkdir(parents=True,exist_ok=True)
    child_output=output_path.with_name(output_path.stem+'.attempt-'+uuid.uuid4().hex+'.json')
    report_path=output_path.parent/'construction-audit-cost.json'
    report={'schema_version':'slim_construction_audit_cost_v1','status':'running',
            'test_id':test_id,'effective_timeout_seconds':timeout,'max_rss_bytes':max_rss_bytes,
            'configured_audit_timeout_seconds':limits.audit_timeout_seconds,
            'native_simulation_timeout_seconds':limits.simulation_timeout_seconds,
            'trace_file':str(trace_path),'prepared_scene_file':str(prepared_scene_path),
            'audit_file':str(output_path),'child_audit_file':str(child_output),
            'sample_policy':'all recorded native solver samples; no thinning',
            'scope':'Separate source-motion audit cap; not a combined native-simulation-plus-audit timeout.'}
    write_json_atomic(report_path,report)
    command=[str(cad_python or sys.executable),'-X','utf8','-m',
             'simulation_ready_cad_workflow.validation_pipeline.slim_harness.construction_audit',
             str(prepared_scene_path),str(trace_path),str(child_output)]
    child_env=dict(os.environ)
    if cwd is not None:
        child_env['PYTHONPATH']=str(Path(cwd).resolve())+os.pathsep+child_env.get('PYTHONPATH','')
    started=time.monotonic()

    def incomplete(code,message,**details):
        failure=ConstructionLimitError(code,message,test_id=test_id,trace_file=str(trace_path),
            audit_file=str(output_path),cost_report=str(report_path),evidence_status='incomplete',
            failure_cause='not_uniquely_determined',effective_timeout_seconds=timeout,**details)
        # Explicitly replace any stale success at the canonical report path.
        # The completed physical trace and original inputs are never rewritten.
        write_json_atomic(output_path,{'schema_version':'slim_construction_audit_incomplete_v1',
            'status':'evidence_incomplete','passed':False,'failure':failure.details,
            'trace_file':str(trace_path),'checked_solver_samples':None,
            'sample_policy':report['sample_policy'],'elapsed_seconds':time.monotonic()-started})
        return failure

    try:
        execution=run_bounded_subprocess(command,cwd=cwd,env=child_env,
            timeout_seconds=timeout,max_rss_bytes=max_rss_bytes)
        (output_path.parent/'geometry-audit.stdout.log').write_bytes(execution.stdout)
        (output_path.parent/'geometry-audit.stderr.log').write_bytes(execution.stderr)
        report['returncode']=execution.returncode
        report['memory_budget']=getattr(execution,'memory_budget',None)
        if execution.returncode:
            report['status']='process_failed'
            raise incomplete('CONSTRUCTION_AUDIT_PROCESS_FAILED',
                'Original-geometry audit did not finish successfully. Inspect its process log; preserve the native trace and choose a repair strategy.',
                returncode=execution.returncode)
        try:
            audit=json.loads(child_output.read_text(encoding='utf-8'))
            if not isinstance(audit,dict) or not isinstance(audit.get('passed'),bool):
                raise ValueError('Missing auditor verdict')
            if audit['passed'] and (not isinstance(audit.get('available_solver_samples'),int)
                    or audit['available_solver_samples']<1
                    or audit.get('checked_solver_samples')!=audit['available_solver_samples']):
                raise ValueError('A successful original-geometry audit must cover every recorded solver sample')
        except (OSError,ValueError) as exc:
            report['status']='invalid_output'
            raise incomplete('CONSTRUCTION_AUDIT_INCOMPLETE',
                f'No complete source-motion audit result was produced: {exc}') from exc
        write_json_atomic(output_path,audit)
        report.update(status='finished',audit_passed=audit['passed'],
                      checked_solver_samples=audit.get('checked_solver_samples'),
                      available_solver_samples=audit.get('available_solver_samples'))
        return audit
    except BoundedSubprocessError as exc:
        timed_out=exc.code=='BOUNDED_SUBPROCESS_TIMEOUT'
        report.update(status='timeout' if timed_out else 'resource_error',subprocess_error=exc.code,memory_budget=getattr(exc,'memory_budget',None))
        raise incomplete('CONSTRUCTION_AUDIT_HARD_TIMEOUT' if timed_out else 'CONSTRUCTION_AUDIT_RESOURCE_LIMIT',
            'Original-geometry audit stopped at its resource limit; physical trace remains available, but geometry acceptance is incomplete. '
            'Choose mechanical repair, scoped simulation simplification with static geometry plus flexible/abstract joints, '
            'or a non-core test adjustment. No sample was silently omitted and no physical PASS is granted.',
            subprocess_error=exc.code) from exc
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        write_json_atomic(report_path,report)


def execute_motion_audit(prepared_scene_path,trace_path,output_path):
    """Child entrypoint: pass the complete stored sequence to the existing audit."""
    from .audit import audit_motion
    from .trace_storage import open_trace,release_trace_cache
    prepared=json.loads(Path(prepared_scene_path).read_text(encoding='utf-8'))
    trace=open_trace(Path(trace_path))
    try:
        return audit_motion(prepared,trace['samples'],Path(output_path))
    finally:
        release_trace_cache(trace)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Bounded construction original-geometry audit child')
    parser.add_argument('prepared_scene',type=Path)
    parser.add_argument('trace',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args(argv)
    result=execute_motion_audit(args.prepared_scene,args.trace,args.output)
    print(json.dumps({key:result.get(key) for key in ('passed','checked_solver_samples','available_solver_samples')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
