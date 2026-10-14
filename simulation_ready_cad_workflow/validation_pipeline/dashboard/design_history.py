"""Read history and request only the preview the user is viewing."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from ..trajectory.store import HistoryStore, project_from_path
from . import history_preview
from .preview_service import _service


def store_for(run):
    project=project_from_path(run)
    if project is None:raise ValueError('Run has no project history')
    return HistoryStore(project)


def frame(service,store,run_id,snapshot):
    path=history_preview.model_path(service,store.project,snapshot)
    at=snapshot.get('at_ms')
    key='history:'+str(store.project)+':'+snapshot['id']
    record=service.records.get(key,{})
    return {k:snapshot.get(k) for k in ('id','seq','revision','turn','event_seq','capture','status','source_complete','command','error','observation_note','up_axis','geometry_ready_ms','completed_ms')} | {
        'run_id':run_id, 'created_at':datetime.fromtimestamp(at/1000,timezone.utc).isoformat() if at is not None else None,
        'model_url':f'/api/preview-file?run={quote(run_id,safe="")}&name=history/{quote(snapshot["id"],safe="")}' if path else None,
        'preview_status':'ready' if path else record.get('status','missing'),
        'preview_error':record.get('error'), 'bytes':path.stat().st_size if path else 0,
        'source_file_count':len(snapshot.get('files',{}))}


def history_view(workspace,run_id,run,*,around=None,selected=None,before=None,scope="revision"):
    if scope=="project":
        from .project_history import project_history_view
        return project_history_view(workspace,run_id,run,around=around,selected=selected,before=before)
    if scope!="revision":raise ValueError("Unknown history scope")
    try:store=store_for(run)
    except ValueError:return {'items':[],'total':0,'offset':0,'latest_seq':0,'indexing':False}
    service=_service(workspace)
    data=store.history(Path(run).name,around=around,before=before,around_id=selected)
    picked=store.snapshot(selected) if selected else (data['items'][-1] if data['items'] else None)
    if picked and picked['revision']!=Path(run).name:raise ValueError('Snapshot belongs to another revision')
    if picked:history_preview.ensure(service,store.project,picked,selected=bool(selected))
    result={k:v for k,v in data.items() if k!='items'}
    result['scope']='revision'
    result['items']=[frame(service,store,run_id,item) for item in data['items']]
    if picked:
        result['selected']=frame(service,store,run_id,picked)
        if not result['selected']['model_url']:
            for previous in store.preceding_snapshots(picked):
                candidate=frame(service,store,run_id,previous)
                if candidate['model_url']:
                    result['display_fallback']=candidate
                    break
    from .history_markers import marker_view
    result['markers']=marker_view(run)
    result['capture']=store.state('capture',{'status':'historical','mode':'recovered'})
    result['storage']={'preview_cache_bytes':getattr(service,'history_cache_bytes',0),
                       'preview_cache_limit':history_preview.CACHE_BYTES,'original_retention':'all',**store.storage()}
    return result


def read_model(workspace,run,snapshot_id):
    store=store_for(run);snapshot=store.snapshot(snapshot_id)
    if not snapshot or snapshot['revision']!=Path(run).name:raise ValueError('Unknown model snapshot')
    path=history_preview.model_path(_service(workspace),store.project,snapshot,touch=True)
    if not path:raise ValueError('History preview is not ready')
    return path
