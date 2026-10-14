"""Small status/activity projections. Mechanical verdicts remain model-owned."""
from pathlib import Path
import re
import time

from .live_workflow import _read_json
from .model import group_runs_by_project
from ..trajectory.store import HistoryStore, project_from_path
from ..trajectory.events import display_text

_REASONS = {
    'provider_error':'作者调用失败', 'ACTIVE_BUDGET_EXHAUSTED':'运行预算已用完',
    'BUDGET_EXHAUSTED':'运行预算已用完', 'RESOURCE_POLICY_REQUIRED':'需要确认运行资源',
    'RESOURCE_APPROVAL_REQUIRED':'需要处理资源请求', 'SIMULATION_TIMEOUT':'仿真超时',
    'NOT_SUBMITTED':'尚未提交完整设计', 'STALE_EVIDENCE':'模型已变化，需要重新验证',
    'physical_task_return_torque_target_not_reached':'回程转矩未达到目标',
    'physical_contact_model_angle_scope_exceeded':'接触模型的角度超出适用范围',
}


def short_reason(run, row=None):
    row=row or {}
    result=_read_json(Path(run)/'result.json') or _read_json(Path(run)/'run_status.json')
    issues=result.get('issues') or []
    if isinstance(issues,str):issues=[issues]
    concrete=[];generic=[]
    for issue in issues:
        message=issue.get('message') or issue.get('detail') or issue.get('code') if isinstance(issue,dict) else str(issue)
        if not message:continue
        code=issue.get('code') if isinstance(issue,dict) else ''
        (generic if code in {'LOAD_LANE_FAILED','TEST_LANE_FAILED'} else concrete).append(message)
    workflow=row.get('live_workflow') or {}
    failed_tests=[]
    for test in result.get('tests') or []:
        if not isinstance(test,dict):continue
        for assertion in test.get('assertions') or []:
            if isinstance(assertion,dict) and assertion.get('passed') is False:
                reason=assertion.get('purpose') or assertion.get('id')
                if reason:failed_tests.append(str(test.get('test_id') or test.get('id') or '测试')+'：'+str(reason)+'未通过')
        if test.get('failure') or test.get('reason'):
            failed_tests.append(str(test.get('test_id') or '测试')+'：'+str(test.get('failure') or test.get('reason')))
    message=next(iter(concrete or failed_tests or generic),None)
    if not message:
        message=workflow.get('stop_reason') or result.get('reason') or result.get('reason_code')
    if not message:
        message=row.get('terminal_reason') if row.get('status') in {'failed','blocked'} else ''
    if not message:return '原因未记录' if row.get('status') in {'failed','blocked'} else ''
    for key,value in _REASONS.items():message=str(message).replace(key,value)
    return re.sub(r'\s+',' ',display_text(str(message))).strip()[:160]


def activity_view(project, workflow=None):
    workflow=workflow or {}
    store=HistoryStore(project)
    turn=(workflow.get('active_turn') or {}).get('id')
    with store.connect() as db:
        if db is None:return {'available':False,'summary':'暂无公开活动记录'}
        where=' WHERE turn=?' if turn else ''
        row=db.execute('SELECT seq,turn,at_ms,payload FROM events'+where+' ORDER BY seq DESC LIMIT 1',(turn,) if turn else ()).fetchone()
        bound=db.execute('SELECT data FROM snapshots WHERE event_seq=? ORDER BY seq DESC LIMIT 1',(row['seq'],)).fetchone() if row else None
    if not row:return {'available':False,'summary':'暂无公开活动记录'}
    import json
    item=json.loads(row['payload'])[0]
    text=item.get('text','');title=item.get('title') or '公开消息'
    running=item.get('status') in {'running','in_progress','started'} and bool(workflow.get('is_running'))
    summary=title
    # label/detail split "label：detail" summaries so the page can leave the
    # agent-written detail untranslated.
    label=detail=None
    attempt=json.loads(bound[0]) if bound else {}
    verb={'build':'构建模型','preview':'生成预览','check':'检查模型','simulate':'运行仿真','probe':'仿真探测'}.get(attempt.get('command'))
    if item.get('kind')=='tool':
        if title=='修改文件' or any(name in title.lower() for name in ('write_file','edit_file')):
            match=re.search(r'[\w./-]+\.(?:py|json|toml|yaml|scad)',text)
            summary='编辑 '+(Path(match[0]).name if match else '文件')
        elif verb:summary=verb+('失败' if 'FAIL' in str(attempt.get('status')) else '')
        elif title=='运行命令':label,detail='运行命令',text.splitlines()[0][:75]
    elif item.get('kind')=='message':label,detail='消息',re.sub(r'\s+',' ',text)[:85]
    elif item.get('kind')=='status':summary=title
    elif item.get('kind')=='error':label,detail='执行错误',text[:80]
    if detail is not None:summary=label+'：'+detail
    view={'available':True,'summary':display_text(summary),'at_ms':row['at_ms'],'event_seq':row['seq'],
          'turn':row['turn'],'running':running}
    if detail is not None:view.update(label=label,detail=display_text(detail))
    return view


def project_summary(server):
    rows=server.runs()
    projects=group_runs_by_project(rows)
    by_id={r['id']:r for r in rows}
    output=[]
    for project in projects:
        run_id=project.get('current_run_id') or project.get('default_run_id') or project.get('latest_run_id')
        row=by_id.get(run_id,{})
        if not row:continue
        run=server.workspace_root/run_id
        workflow=row.get('live_workflow') or {}
        status=row.get('effective_status') or row.get('status') or 'not_started'
        active=bool(workflow.get('is_running') or row.get('healthy_running') or project.get('active_status')=='running')
        if active:status='running'
        # An intermediate negative revision is not a terminal project result.
        passed=status=='passed' and bool(row.get('eligible_pass'))
        stopped=not active and status in {'failed','blocked'}
        attention=bool(workflow.get('stop_reason')) or row.get('terminal_reason') not in {None,'','not_submitted','NOT_SUBMITTED'}
        notify=not active and (passed or stopped and (status=='failed' or attention))
        reason=short_reason(run,row)
        if active:
            tests=workflow.get('test_progress') or {}
            reason=('正在运行：'+tests['active_test']) if tests.get('active_test') else '执行中；历史失败记录保留在版本证据中'
        output.append({'id':project['id'],'title':project.get('title') or project['id'],'run_id':run_id,
            'revision':row.get('revision'),'status':status,'running':active,'reason':reason,
            'notify':bool(notify),'notice_key':f"{project['id']}|{run_id}|{status}|{workflow.get('stop_reason') or ''}|{workflow.get('ended_at') or row.get('updated_at') or ''}",
            'updated_at':row.get('updated_at'),
            'versions':[{'id':r['id'],'reason':short_reason(server.workspace_root/r['id'],r)} for r in project.get('runs',[])]})
    return {'projects':output,'ready':server.catalog_state()['ready'],'server_time_ms':int(time.time()*1000)}
