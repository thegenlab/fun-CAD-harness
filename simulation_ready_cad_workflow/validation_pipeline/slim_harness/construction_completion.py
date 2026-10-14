"""Always publish evidence-based partial completion, even after timeout/crash."""
from __future__ import annotations

import json
import hashlib
import copy
from pathlib import Path
from datetime import datetime, timezone

from ..mechanism.io import write_json_atomic
from ..cold_start.token_watch import normalize_token_usage
from .construction_budget import ProjectBudget


def _read(path,default=None):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return value if isinstance(value,dict) else {} if default is None else default
    except (OSError,ValueError,TypeError):return {} if default is None else default


def _digest(path):
    try:
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    except (OSError,TypeError):return None


def _modified(path):
    try:return Path(path).stat().st_mtime_ns
    except OSError:return 0


def _event_usage(path):
    try:
        lines=Path(path).read_text(encoding='utf-8',errors='replace').splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if '"usage"' not in line or ('"type":"result"' not in line and '"type": "result"' not in line
                                     and '"turn.completed"' not in line):
            continue
        try:
            event=json.loads(line)
        except (ValueError,TypeError):
            continue
        if not isinstance(event,dict) or event.get('type') not in {'result','turn.completed'}:
            continue
        usage=normalize_token_usage(event.get('usage'))
        if usage:
            return usage
    return None


def _formal_cli_started(path,client,session_id):
    """Require original CLI events, rather than a controller/preflight receipt."""
    try:
        with Path(path).open(encoding='utf-8',errors='replace') as stream:
            for line in stream:
                try:
                    event=json.loads(line)
                except (ValueError,TypeError):
                    continue
                if not isinstance(event,dict):continue
                event=event.get('event',event)
                if not isinstance(event,dict):continue
                kind=event.get('type')
                if client=='codex' and kind in {'thread.started','turn.started','turn.completed','item.started','item.completed'}:
                    observed=event.get('thread_id')
                    if kind=='thread.started':
                        if isinstance(observed,str) and observed:return not session_id or observed==session_id
                    else:return True
                if client=='cursor' and (kind in {'assistant','tool_call','result'}
                        or kind=='system' and event.get('subtype')=='init'):
                    observed=event.get('session_id')
                    if isinstance(observed,str) and observed:return not session_id or observed==session_id
    except OSError:
        pass
    return False


def _formal_cli_usage(turn,receipt_path):
    receipt=_read(receipt_path)
    if receipt.get('schema_version')!=1 or receipt.get('client') not in {'codex','cursor'}:
        return None
    if not _formal_cli_started(turn/'events.jsonl',receipt['client'],receipt.get('session_id')):
        return None
    response=_read(turn/'response.json')
    usage=normalize_token_usage(response.get('usage')) or _event_usage(turn/'events.jsonl')
    return {'call_id':receipt.get('call_id') or turn.name,'usage':usage,
        'status':'completed' if receipt.get('client_completed') is True else receipt.get('status','incomplete'),
        'record_path':str(receipt_path)}


def _model_usage(project):
    """Live usage files are per CLI call, including interrupted calls, not
    cumulative persistent-session counters. Cached input is a subset of input.
    Formal turns without a live snapshot use their own reported CLI usage.
    Native session IDs never merge separate CLI invocations.
    """
    calls={};observed_turns=set()
    for path in (project/'worker_turns').glob('*/usage.live.json'):
        record=_read(path)
        identifier=record.get('call_id') or path.parent.name
        usage=normalize_token_usage(record.get('usage')) or _event_usage(path.with_name('events.jsonl'))
        calls[identifier]={**record,'usage':usage,'record_path':str(path)}
        observed_turns.add(path.parent)
    for path in (project/'worker_turns').glob('*/.benchmark-sandbox/sandbox-run/formal-receipt.json'):
        turn=path.parents[2]
        if turn in observed_turns:continue
        record=_formal_cli_usage(turn,path)
        if record is not None:calls.setdefault(record['call_id'],record)
    known=[r for r in calls.values() if isinstance(r.get('usage'),dict)]
    totals={key:sum(int(r['usage'].get(key,0) or 0) for r in known)
            for key in ('input_tokens','cached_input_tokens','output_tokens')}
    return {'reported_totals':totals,'total_tokens':totals['input_tokens']+totals['output_tokens'],
        'calls':len(calls),'calls_without_reported_usage':len(calls)-len(known),
        'incomplete_call_snapshots':sum(r.get('status') not in {'completed','complete'} for r in calls.values()),
        'scope':'Project worker/reviewer CLI usage only; repository implementation, coordinator conversation and external tool charges are not included. Interrupted calls may have incomplete counters. Cached input is already included in input.'}


def _geometry_matches(geometry,scene,scene_path):
    """Reused evidence must match both current scene and actual CAD bytes.

    Author tools reserialize JSON into their snapshot, so use content equality
    after checking that snapshot's existing byte binding; whitespace is not CAD.
    """
    snapshot=geometry.get('scene_file')
    if not snapshot or not geometry.get('scene_sha256'):return False
    if _digest(snapshot)!=geometry['scene_sha256']:return False
    # Trials are independent. Editing a different test must not erase valid
    # evidence for this one. All shared scene physics still match exactly;
    # _test_evidence separately compares this trial's complete compiled test.
    shared=lambda value:{key:data for key,data in value.items() if key!='tests'}
    if shared(_read(snapshot))!=shared(scene):return False
    bodies={b.get('id'):b for b in geometry.get('bodies',[]) if isinstance(b,dict)}
    expected=scene.get('bodies') or []
    if any(not isinstance(body,dict) or not isinstance(body.get('source'),str) for body in expected):return False
    if not expected or set(bodies)!={b.get('id') for b in expected}:return False
    for body in expected:
        record=bodies[body['id']];digest=record.get('source_sha256')
        if not digest or _digest(scene_path.parent/body.get('source',''))!=digest:return False
        if _digest(record.get('source_file'))!=digest:return False
    return True


def _bound_result(result,scene_path,criteria_path):
    bindings=result.get('source_bindings',{})
    return (bool(bindings) and bindings.get(str(scene_path))==_digest(scene_path)
            and result.get('criteria_sha256')==_digest(criteria_path)
            and all(_digest(path)==value for path,value in bindings.items()))


def _test_evidence(request_path,geometry,tests,formal,scene_units):
    request=_read(request_path);test=request.get('test',{});identifier=test.get('id')
    if identifier not in tests or scene_units not in {'mm','m'}:return None
    expected=copy.deepcopy(tests[identifier])
    # compile_scene changes only loads[].point within tests: source coordinates
    # become metres. Compare that exact compiled representation without running
    # geometry compilation during finalization or relaxing any physical values.
    scale=.001 if scene_units=='mm' else 1.
    try:
        for load in expected.get('loads',[]):
            if 'point' in load:load['point']=[coordinate*scale for coordinate in load['point']]
    except (TypeError,ValueError):return None
    if test!=expected:return None
    trace_path=request_path.parent/'physical.trace.json'
    trace=_read(trace_path) if Path(trace_path).is_file() else {}
    if Path(trace_path).is_file() and trace.get('test_id')!=identifier:return None
    audit_path=request_path.parent/'geometry-audit.json'
    audit=_read(audit_path) if formal else {}
    audit_cost_path=request_path.parent/'construction-audit-cost.json'
    audit_cost=_read(audit_cost_path) if formal else {}
    cost=(_read(request_path.parent/'construction-cost.json')
          or _read(request_path.parent/'construction-simulation.json'))
    scope=(trace.get('validation_scope') or request.get('scene',{}).get('validation_scope')
           or geometry.get('validation_scope') or {'mode':'rigid_original','display_name':'原始刚体模型；证据完整性见逐项状态'})
    finished=Path(trace_path).is_file()
    return {'test_id':identifier,'physical_pass':trace.get('physical_pass') is True,
        'execution_status':'finished' if finished else 'interrupted_or_incomplete',
        'failed_assertion_count':sum(isinstance(a,dict) and a.get('passed') is False for a in trace.get('assertions',[])),
        'geometry_audit_pass':audit.get('passed') is True,
        'geometry_audit_status':audit.get('status') or ('passed' if audit.get('passed') is True else 'failed' if audit else 'not_completed'),
        'geometry_audit_failure':audit.get('failure'),'geometry_audit_violations':audit.get('violations',[]),
        'geometry_audit_file':str(audit_path) if audit else None,
        'geometry_audit_cost':audit_cost,'geometry_audit_cost_file':str(audit_cost_path) if audit_cost else None,
        'native_simulation_cost':cost,
        'failure':trace.get('failure') or (None if finished else cost or 'Execution has a request but no completed physical trace.'),
        'numerical_failure_unresolved':trace.get('numerical_failure_unresolved',False),
        'trace_file':str(trace_path) if finished else None,'request_file':str(request_path),
        'assertions':trace.get('assertions',[]),'scope':scope,
        'evidence_level':'formal_physics' if formal and finished else 'development_physics' if finished else 'execution_attempt',
        '_modified':max(_modified(request_path),_modified(trace_path),_modified(audit_path) if formal else 0)}


def _template_summary(project,run,design,state,budget,files,cad_files,reason,supplied):
    from .publication import read_result
    validated=read_result(run)
    recorded=_read(run/'result.json')
    eligible=(validated.get('template_result') is True and validated.get('eligible_pass') is True
              and validated.get('status')=='PASS'
              and (supplied is None or supplied.get('status')=='PASS'))
    rows=validated.get('tests',recorded.get('tests',[]))
    if not isinstance(rows,list):rows=[]
    rows=[row for row in rows if isinstance(row,dict) and isinstance(row.get('test_id'),str)]
    scope=validated.get('validation_scope')
    limitations=list(scope.get('excluded_claims',[])) if isinstance(scope,dict) else []
    failure=None if eligible else 'Current selected-template publication evidence is missing, stale or incomplete.'
    tasks=[];tests=[]
    for row in rows:
        identifier=row['test_id']
        paths=[row.get(name) for name in ('native_summary_file','native_csv_file','native_poses_file')]
        evidence=[path for path in paths if isinstance(path,str)] if eligible else []
        measured={'test_id':identifier,'physical_pass':eligible and row.get('physical_pass') is True,
            'execution_status':'finished' if eligible else 'unverified_or_incomplete',
            'geometry_audit_pass':eligible and row.get('geometry_audit',{}).get('passed') is True,
            'assertions':row.get('assertions',[]) if eligible else [],'failure':failure,
            'evidence_level':'validated_template_native_physics' if eligible else 'unverified_template_evidence',
            'scope':validated.get('validation_scope') or validated.get('scope'),
            'evidence_paths':evidence}
        tests.append(measured)
        tasks.append({'task_id':identifier,'description':identifier,
            'completion_status':'completed' if eligible else 'partial',
            'design_status':'source_and_template_review_verified' if eligible else 'geometry_present_unverified' if cad_files else 'missing',
            'test_ids':[identifier],'test_status':'verified_within_declared_scope' if eligible else 'evidence_not_currently_valid',
            'executed_tests':[measured] if eligible else [],'verified_complete':eligible,'evidence_paths':evidence,
            'limitations':limitations,
            'reason':'Reviewed template fill, source bindings, complete native trials and source-motion audits passed.' if eligible else failure,
            'recommendations':[] if eligible else ['Inspect the retained template publication evidence; preserve failed or missing records.']})
    if not tasks:
        tasks=[{'task_id':'selected_template_unassessed','description':'Selected template trials',
            'completion_status':'design_only' if cad_files else 'not_started','design_status':'geometry_present_unverified' if cad_files else 'missing',
            'test_ids':[],'test_status':'not_executed','executed_tests':[],'verified_complete':False,'evidence_paths':[],
            'limitations':[],'reason':failure,'recommendations':['Complete the selected template workflow.']}]
    complete=eligible and bool(rows)
    status='COMPLETED' if complete else 'BUDGET_EXHAUSTED_PARTIAL' if budget['work_remaining_seconds']<=0 else 'PARTIAL'
    return {'schema_version':'slim_construction_completion_v1','project_id':project.name,
        'updated_at':datetime.now(timezone.utc).isoformat(),'status':status,
        'stop_reason':reason or state.get('stop_reason') or validated.get('status') or 'Incomplete selected-template execution',
        'budget':budget,'revision':int(state.get('revision',1)),'tasks':tasks,'task_count':len(tasks),
        'verified_completed_tasks':len(tasks) if complete else 0,'tasks_with_executed_tests':len(rows) if eligible else 0,
        'verified_completion_fraction':1. if complete else 0.,'model_usage':_model_usage(project),
        'assets':files,'cad_assets':cad_files,'planned_test_count':len(rows),'test_results':tests,
        'task_denominator_basis':'selected_template_trials','static_geometry_evidence':[],
        'rejected_stale_or_unbound_evidence':[] if complete else [{'path':str(run/'result.json'),'reason':failure}],
        'tool_issues':validated.get('issues',[]),'usability':'verified_benchmark_candidate_with_declared_scope' if complete else 'partial_candidate_requires_review',
        'complete_instance_pass':complete,'core_semantics_status':'selected_template_review_and_native_gates_verified' if complete else 'template_evidence_incomplete',
        'incomplete_reasons':[] if complete else [failure],
        'modeling_and_test_plan':_read(design/'construction_plan.json'),'recommendations':[] if complete else [failure],
        'execution_mode':_read(project/'mode.json').get('mode'),'publication_eligible_pass':eligible,
        'scope':validated.get('scope') or 'Selected-template native workflow; existing publication gates are unchanged.'}


def _uses_selected_template(project):
    return _read(project/'mode.json').get('mode')=='free'


def write_completion_summary(project,*,reason='',result=None):
    project=Path(project).resolve();state=_read(project/'workflow.json');budget=ProjectBudget(project).snapshot()
    revision=int(state.get('revision',1));run=project/'runs'/f'r{revision:02d}';design=run/'design'
    scene_path=design/'scene.json';scene=_read(scene_path);criteria=_read(project/'criteria.json').get('criteria',[])
    results=result if result is not None else _read(run/'result.json');tests={t['id']:t for t in (scene.get('tests') or []) if isinstance(t,dict) and t.get('id')}
    files=[p.relative_to(project).as_posix() for p in sorted(design.rglob('*')) if p.is_file() and p.suffix.lower() in {'.step','.stp','.glb','.stl','.obj','.py','.json'}]
    cad_files=[p for p in files if Path(p).suffix.lower() in {'.step','.stp','.glb','.stl','.obj'}]
    if _uses_selected_template(project):
        return _publish_summary(project,_template_summary(project,run,design,state,budget,files,cad_files,reason,result))
    actual={};static_evidence=[];rejected=[];tool_issues=[]
    def accept(row):
        if not row:return
        current=actual.get(row['test_id'])
        if current is None:
            actual[row['test_id']]=row;return
        current_pass=current.get('physical_pass') and current.get('execution_status')=='finished'
        incoming_pass=row.get('physical_pass') and row.get('execution_status')=='finished'
        # Formal exhaust writes a later request/cost file. That incomplete
        # attempt must not erase a finished development or formal physical pass.
        if current_pass and row.get('execution_status')!='finished':return
        if incoming_pass and not current_pass:
            actual[row['test_id']]=row;return
        if row['_modified']>=current['_modified']:actual[row['test_id']]=row
    # Read actual current-design records in order. A later failure must remain
    # visible even if an earlier execution of the same test passed.
    for geometry_path in sorted((run/'attempts').glob('a*/geometry/geometry.json')):
        geometry=_read(geometry_path)
        if not _geometry_matches(geometry,scene,scene_path):
            rejected.append({'path':str(geometry_path),'reason':'Scene or source geometry differs from current design.'});continue
        attempt=geometry_path.parent.parent;initial_path=attempt/'initial-geometry-audit.json';initial=_read(initial_path)
        if initial:static_evidence.append({'path':str(initial_path),'passed':initial.get('passed') is True,'scope':'original_geometry_initial_pose_only','violations':initial.get('violations',[])})
        for request_path in sorted((attempt/'tests').glob('*/request.json')):accept(_test_evidence(request_path,geometry,tests,True,scene.get('units')))
    for p in sorted((project/'construction-evidence').glob('*.json'),key=_modified):
        record=_read(p);report=_read(record.get('result_file'));geometry=_read(report.get('geometry'))
        bound=_geometry_matches(geometry,scene,scene_path)
        if report.get('issues'):tool_issues.append({'record':str(p),'issues':report['issues'],'current_design_verified':bound})
        if not bound:
            rejected.append({'path':str(p),'reason':'Missing or changed current-scene/source binding.'});continue
        initial=_read(report.get('initial_audit'))
        if initial:static_evidence.append({'path':report['initial_audit'],'passed':initial.get('passed') is True,'scope':'original_geometry_initial_pose_only','violations':initial.get('violations',[])})
        for request_path in sorted((Path(record['result_file']).parent/'tests').glob('*/request.json')):accept(_test_evidence(request_path,geometry,tests,False,scene.get('units')))
    for row in actual.values():row.pop('_modified',None)
    bound_result=_bound_result(results,scene_path,project/'criteria.json')
    final_review=results.get('review',{}) if bound_result else {}
    observations={r['criterion_id']:r for r in final_review.get('observations',[]) if isinstance(r,dict) and r.get('criterion_id')}
    coverage={r['criterion_id']:r.get('test_ids',[]) for r in final_review.get('test_coverage',[]) if isinstance(r,dict) and r.get('criterion_id')}
    functions=[r for r in criteria if isinstance(r,dict) and r.get('kind')=='function' and r.get('id')]
    denominator_basis='independent_core_criteria'
    if not functions:
        functions=[{'id':t,'statement':data.get('purpose',t)} for t,data in tests.items()]
        denominator_basis='unreviewed_scene_tests'
    if not functions:
        task_text=(project/'task.md').read_text(encoding='utf-8') if (project/'task.md').is_file() else 'Mechanical asset and core functional tests'
        functions=[{'id':'core_task_unassessed','statement':task_text.strip()}]
        denominator_basis='core_task_not_yet_decomposed'
    tasks=[]
    for criterion in functions:
        identifier=criterion['id'];matching=[tid for tid,t in tests.items() if any(isinstance(a,dict) and identifier in (a.get('criterion_ids') or []) for a in (t.get('assertions') or []))]
        matching=list(dict.fromkeys(matching+[tid for tid in coverage.get(identifier,[]) if tid in tests]))
        if not matching and identifier in tests:matching=[identifier]
        measured=[actual[t] for t in matching if t in actual]
        passed=[t for t in measured if t.get('physical_pass') and not t.get('numerical_failure_unresolved')]
        observed=observations.get(identifier,{})
        verified=bool(matching) and len(passed)==len(matching) and all(t.get('geometry_audit_pass') for t in passed) and observed.get('verdict')=='met'
        if verified:status='completed'
        elif measured:status='partial' if passed or any(t['execution_status']!='finished' for t in measured) else 'failed'
        elif files:status='design_only'
        else:status='not_started'
        reasons=[]
        if not matching:reasons.append('Core task has no linked test definition yet.')
        elif not measured:reasons.append('No executed test evidence for the current design.')
        for row in measured:
            if row['failure']:reasons.append(f"{row['test_id']}: {row['failure']}")
            failed=[a for a in row.get('assertions',[]) if isinstance(a,dict) and a.get('passed') is False]
            if failed:
                # A solver can finish normally yet fail the task. Preserve the
                # actual failed conditions instead of blaming a missing review.
                examples='; '.join(f"{a.get('id','assertion')}: {a.get('purpose') or a.get('expression') or 'condition not met'}" for a in failed[:5])
                reasons.append(f"{row['test_id']}: {len(failed)} task assertions failed. {examples}")
            if row['numerical_failure_unresolved']:reasons.append(f"{row['test_id']}: numerical failure remains unresolved.")
            if not row['geometry_audit_pass']:
                failure=row.get('geometry_audit_failure')
                if isinstance(failure,dict):failure=' '.join(str(failure.get(key,'')) for key in ('code','message')).strip()
                detail=failure or row.get('geometry_audit_violations') or 'full source-motion audit not complete'
                reasons.append(f"{row['test_id']}: geometry audit {row['geometry_audit_status']}: {detail}.")
        if observed.get('verdict')!='met':reasons.append('Independent final review is incomplete for this design.')
        why='All linked tests, source-motion audits and final review passed within the disclosed model scope.' if verified else ' '.join(reasons)
        test_status=('verified_within_declared_scope' if verified else 'physics_passed_review_or_audit_pending' if passed and len(passed)==len(matching)
                     else 'partially_executed' if any(t['execution_status']!='finished' for t in measured) else 'failed' if measured else 'not_executed')
        tasks.append({'task_id':identifier,'description':criterion.get('statement',identifier),'completion_status':status,
            'design_status':'reviewed' if observed.get('verdict')=='met' else 'geometry_present_unreviewed' if cad_files else 'source_only' if files else 'missing',
            'test_ids':matching,'test_status':test_status,'executed_tests':measured,'verified_complete':verified,
            'evidence_paths':list(dict.fromkeys(p for row in measured for p in (row.get('trace_file'),row.get('request_file'),row.get('geometry_audit_file'),row.get('geometry_audit_cost_file')) if p)),
            'limitations':list(dict.fromkeys(str(claim) for row in measured for claim in (row.get('scope',{}).get('excluded_claims',[]) if isinstance(row.get('scope'),dict) else []))),
            'reason':why,'recommendations':[] if verified else ['Repair mechanical issues, simplify the simulation, or revise non-core test strictness; retain core function and disclose scope.']})
    done=sum(t['verified_complete'] for t in tasks);tested=sum(bool(t['executed_tests']) for t in tasks)
    complete=bool(tasks) and done==len(tasks) and results.get('status')=='PASS' and bound_result
    status='COMPLETED' if complete else 'BUDGET_EXHAUSTED_PARTIAL' if budget['work_remaining_seconds']<=0 else 'PARTIAL'
    summary={'schema_version':'slim_construction_completion_v1','project_id':project.name,'updated_at':datetime.now(timezone.utc).isoformat(),
        'status':status,'stop_reason':reason or state.get('stop_reason') or results.get('status') or 'Incomplete execution',
        'budget':budget,'revision':revision,'tasks':tasks,'task_count':len(tasks),'verified_completed_tasks':done,
        'model_usage':_model_usage(project),
        'tasks_with_executed_tests':tested,'verified_completion_fraction':done/len(tasks) if tasks else 0.,
        'assets':files,'cad_assets':cad_files,'planned_test_count':len(tests),'test_results':list(actual.values()),
        'task_denominator_basis':denominator_basis,'static_geometry_evidence':static_evidence,
        'rejected_stale_or_unbound_evidence':rejected,'tool_issues':tool_issues,
        'usability':'verified_benchmark_candidate_with_declared_scope' if complete else 'partial_candidate_requires_review' if cad_files else 'source_only_no_cad_asset_yet' if files else 'no_asset_yet',
        'complete_instance_pass':complete,'core_semantics_status':'independently_reviewed' if complete else 'not_yet_fully_reviewed',
        'incomplete_reasons':[] if complete else [reason or state.get('stop_reason') or results.get('status') or 'Incomplete execution',*results.get('issues',[])],
        'modeling_and_test_plan':_read(design/'construction_plan.json'),
        'recommendations':[] if complete else ['Inspect per-task missing evidence. Prioritize core behavior; use flexible/abstract joint models with original static geometry checks if detailed contact exceeds the time budget.'],
        'scope':'Benchmark construction completion within disclosed model scope. Static checks, development physics and independently reviewed results are separate; no frozen evaluation certificate.'}
    return _publish_summary(project,summary)


def _publish_summary(project,summary):
    budget=summary['budget'];tasks=summary['tasks'];files=summary['assets'];status=summary['status']
    done=summary['verified_completed_tasks'];tested=summary['tasks_with_executed_tests']
    tests=range(summary['planned_test_count'])
    write_json_atomic(project/'completion_summary.json',summary)
    usage=summary['model_usage'];totals=usage.get('reported_totals') or {}
    lines=[f'# {project.name} — 构建完成度总结','',f'状态：{status}。终止原因：{summary["stop_reason"]}',
        f'有效工作：{budget["active_seconds"]/60:.2f} / {budget["total_seconds"]/60:.0f} 分钟；不计入预算的仿真/等待/安装：{budget["excluded_seconds"]/60:.2f} 分钟。',
        f'已验证完成：{done}/{len(tasks)}；已有执行证据：{tested}/{len(tasks)}；计划测试：{len(tests)}。',
        f'模型用量：input {int(totals.get("input_tokens") or 0)} / output {int(totals.get("output_tokens") or 0)} / 合计 {int(usage.get("total_tokens") or 0)}（{int(usage.get("calls") or 0)} 次调用，{int(usage.get("calls_without_reported_usage") or 0)} 次未报）。','',
        '| 任务 | 完成度 | 设计 | 测试与未完成原因 |','|---|---|---|---|']
    for t in tasks:
        esc=lambda s:str(s).replace('|','/').replace('\n',' ')
        lines.append('| '+' | '.join(esc(v) for v in (t['description'],t['completion_status'],t['design_status'],t['reason']))+' |')
    lines+=['','## 证据与资产','',*[f'- `{p}`' for p in files[:60]],'','## 后续措施','',*summary['recommendations'],
        '', '测试的近似范围、失败和逐项执行证据见同目录 completion_summary.json。部分完成不等于物理验收通过。']
    (project/'completion_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
