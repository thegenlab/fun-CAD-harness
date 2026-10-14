"""Project-wide pagination over index metadata and existing preview references."""
from datetime import datetime
from pathlib import Path
import re
from urllib.parse import quote

from .design_history import frame,store_for
from . import history_preview
from .history_markers import marker_view
from .path_resolution import resolve_workspace_reference
from .preview_service import _service,_cache_root,preview_timeline


def _recorded_time(value):
    try:
        timestamp=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return int(timestamp.timestamp()*1000) if timestamp.tzinfo else None
    except ValueError:return None


def project_history_view(workspace,run_id,run,*,around=None,selected=None,before=None):
    workspace=Path(workspace).resolve();run=Path(run).resolve()
    store=store_for(run);service=_service(workspace)
    # Read small ordering/reference columns; full source bindings and log bodies
    # are fetched only for the selected page/model.
    with store.connect() as db:
        headers=[dict(row) for row in db.execute('SELECT id,revision,seq,event_seq,at_ms,model,status FROM snapshots')] if db else []
        latest_seq=db.execute('SELECT COALESCE(MAX(seq),0) FROM events').fetchone()[0] if db else 0
    by_revision={}
    for header in headers:by_revision.setdefault(header['revision'],[]).append(header)
    runs=sorted((p for p in (store.project/'runs').iterdir() if p.is_dir() and re.fullmatch(r'r\d+',p.name)),key=lambda p:(int(p.name[1:]),p.name))
    segments=[];references=[]
    for revision_run in runs:
        revision=revision_run.name;own_run=revision_run.relative_to(workspace).as_posix()
        indexed=by_revision.get(revision,[])
        known_models={str(resolve_workspace_reference(workspace,item['model'])) for item in indexed if item.get('model')}
        records=[{**item,'kind':'indexed','run_id':own_run,'order':(item['at_ms'] is not None,item['at_ms'] or 0,item['event_seq'] or item['seq'],0,item['seq'])} for item in indexed]
        for number,item in enumerate(preview_timeline(workspace_root=workspace,run_dir=revision_run)):
            model=(_cache_root(workspace,revision_run)/item['model']).resolve()
            # Identical file references prove a duplicate; matching dates or
            # the presence of another indexed attempt do not.
            if str(model) in known_models:continue
            at=_recorded_time(item.get('created_at'))
            legacy={**item,'id':f'legacy:{revision}:{item["id"]}','legacy_id':item['id'],
                'run_id':own_run,'revision':revision,'capture':'legacy','status':'recovered','preview_status':'ready',
                'model_url':f'/api/preview-file?run={quote(own_run,safe="")}&name={quote("timeline/"+item["model"],safe="")}',
                'source_file_count':0,'source_complete':False}
            records.append({'id':legacy['id'],'kind':'legacy','revision':revision,'run_id':own_run,'frame':legacy,'order':(at is not None,at or 0,number,1,number)})
        records.sort(key=lambda item:item['order'])
        offset=len(references)
        segments.append({'revision':revision,'run_id':own_run,'offset':offset,'count':len(records),
                         'indexed_count':len(indexed),'legacy_count':len(records)-len(indexed)})
        for index,item in enumerate(records):item['ordinal']=offset+index
        references.extend(records)
    total=len(references);by_id={item['id']:item for item in references}
    if selected and selected not in by_id:raise ValueError('Unknown project snapshot')
    if around is None and selected:around=by_id[selected]['ordinal']
    limit=100
    offset=max(0,total-limit) if around is None else max(0,min(total-limit,int(around)-limit//2))
    if before is not None:offset=max(0,int(before)-limit)
    page=references[offset:offset+limit]
    selected_ref=by_id[selected] if selected else page[-1] if page else None
    needed={item['id'] for item in page if item['kind']=='indexed'}
    if selected_ref and selected_ref['kind']=='indexed':needed.add(selected_ref['id'])
    snapshots={}
    if needed:
        with store.connect() as db:
            snapshots={row['id']:store.snapshot_row(row) for row in db.execute('SELECT * FROM snapshots WHERE id IN ('+','.join('?' for _ in needed)+')',tuple(needed))}
    def view(reference):
        if reference['kind']=='legacy':return {**reference['frame'],'ordinal':reference['ordinal']}
        snapshot=snapshots.get(reference['id']) or store.snapshot(reference['id'])
        return {**frame(service,store,reference['run_id'],snapshot),'ordinal':reference['ordinal']}
    if selected_ref and selected_ref['kind']=='indexed':
        history_preview.ensure(service,store.project,snapshots[selected_ref['id']],selected=bool(selected))
    result={'scope':'project','items':[view(item) for item in page],'total':total,'offset':offset,
            'latest_seq':latest_seq,'indexing':False,'revisions':segments,'selected':view(selected_ref) if selected_ref else None}
    if selected_ref and not result['selected'].get('model_url'):
        for previous in reversed(references[:selected_ref['ordinal']]):
            if previous['kind']=='legacy' or previous.get('status')!='incomplete' and history_preview.model_path(service,store.project,previous):
                result['display_fallback']=view(previous);break
    markers=[]
    for segment in segments:
        for marker in marker_view(workspace/segment['run_id']):
            reference=by_id.get(marker.get('snapshot_id'))
            markers.append({**marker,'id':segment['revision']+':'+marker['id'],'revision':segment['revision'],'run_id':segment['run_id'],
                'ordinal':reference['ordinal'] if reference else None})
    result['markers']=markers
    result['capture']=store.state('capture',{'status':'historical','mode':'recovered'})
    result['storage']={'preview_cache_bytes':getattr(service,'history_cache_bytes',0),'preview_cache_limit':history_preview.CACHE_BYTES,
                       'original_retention':'all',**store.storage()}
    return result
