"""Markers express recorded checks; they do not infer why a model failed."""
import json
import time

_CACHE={}
from pathlib import Path

from .path_resolution import resolve_workspace_reference
from .live_workflow import _read_json
from ..trajectory.store import HistoryStore,project_from_path


def marker_view(run):
    run=Path(run);project=project_from_path(run)
    if project is None:return []
    cached=_CACHE.get(str(run))
    if cached and time.monotonic()-cached[0]<2:return cached[1]
    store=HistoryStore(project);markers=[]
    with store.connect() as db:
        rows=db.execute("SELECT id,seq,event_seq,at_ms,status,json_extract(data,'$.command') AS command,json_extract(data,'$.completed_ms') AS completed,json_extract(data,'$.model_binding') AS binding,json_extract(data,'$.test_ids') AS test_ids FROM snapshots WHERE revision=? ORDER BY seq",(run.name,)).fetchall() if db else []
    for row in rows:
        command=row['command'];state=str(row['status'])
        if command=='check' and row['completed'] is not None:
            markers.append({'id':row['id']+':check','snapshot_id':row['id'],'event_seq':row['event_seq'],
                'kind':'check_failed' if state!='CHECKED' else 'check_completed','label':'开发检查失败' if state!='CHECKED' else '开发检查完成','at_ms':row['completed']})
        elif command in {'simulate','probe'} and row['completed'] is not None and (json.loads(row['test_ids'] or '[]') or state=='COST_PROBE_COMPLETE'):
            markers.append({'id':row['id']+':simulation','snapshot_id':row['id'],'event_seq':row['event_seq'],
                'kind':'simulation_result','label':'开发仿真返回 · '+state,'at_ms':row['completed']})
        elif command=='build' and 'FAIL' in state:
            markers.append({'id':row['id']+':build','snapshot_id':row['id'],'event_seq':row['event_seq'],'kind':'build_failed','label':'构建失败','at_ms':row['completed']})
    seen=set();by_binding={};requested={}
    for row in rows:
        if row['binding']:by_binding.setdefault(json.dumps(json.loads(row['binding']),sort_keys=True),[]).append(row)
    for candidates in by_binding.values():candidates.sort(key=lambda item:item['at_ms'] or 0,reverse=True)
    for reference in (project/'trajectory/configurations').glob('*.json'):
        config_ref=_read_json(reference)
        if config_ref.get('revision')!=run.name:continue
        config=_read_json(resolve_workspace_reference(project.parents[2],config_ref.get('file','')))
        identity=str(config.get('attempt',''))+':'+str(config.get('kind'))
        if not identity or identity in seen:continue
        seen.add(identity)
        snapshot_id=config.get('snapshot_id')
        binding=config.get('history_binding') or {}
        if not snapshot_id and binding.get('sources'):
            candidates=by_binding.get(json.dumps(binding,sort_keys=True),[])
            snapshot_id=next((c['id'] for c in candidates if c['at_ms'] is not None and c['at_ms']<=config.get('captured_ms',0)),None)
        if snapshot_id:requested.setdefault(snapshot_id,set()).add(config.get('test_id'))
        markers.append({'id':'simulation:'+identity,'snapshot_id':snapshot_id,'test_id':config.get('test_id'),
            'kind':'simulation_started' if config.get('kind')=='effective' else 'simulation_requested','label':'仿真尝试开始' if config.get('kind')=='effective' else '请求仿真','at_ms':config.get('captured_ms'),'scope':'snapshot' if snapshot_id else 'revision'})
    recorded_results={marker['snapshot_id'] for marker in markers if marker['kind']=='simulation_result'}
    for row in rows:
        if row['command'] in {'simulate','probe'} and row['completed'] is not None and row['id'] in requested and row['id'] not in recorded_results:
            markers.append({'id':row['id']+':simulation','snapshot_id':row['id'],'event_seq':row['event_seq'],
                'test_ids':sorted(t for t in requested[row['id']] if t),'kind':'simulation_failed','label':'开发仿真结束 · '+str(row['status']),'at_ms':row['completed']})
    result=_read_json(run/'result.json') or _read_json(run/'run_status.json')
    if result.get('status'):
        markers.append({'id':'revision-result','snapshot_id':None,'kind':'validation_result','scope':'revision',
            'label':'版本级验证结果 · '+str(result['status']),'at_ms':None})
    ordered=sorted(rows,key=lambda row:(row['at_ms'] is not None,row['at_ms'] or 0,row['event_seq'] or row['seq'],row['seq']))
    ordinals={row['id']:i for i,row in enumerate(ordered)}
    for marker in markers:marker['ordinal']=ordinals.get(marker.get('snapshot_id'))
    _CACHE[str(run)]=(time.monotonic(),markers)
    return markers
