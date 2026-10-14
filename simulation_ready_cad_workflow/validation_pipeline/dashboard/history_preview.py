"""Bounded, disposable previews of durable history snapshots."""
import os
from pathlib import Path
import shutil
import subprocess
import time

from ..trajectory.capture import read_json, write_json
from .path_resolution import resolve_workspace_reference

CACHE_BYTES = 2 * 1024**3


def cache_root(service):
    return service.workspace / '.runtime_runs/dashboard_previews/history_cache'


def cache_dir(service, project, snapshot):
    return cache_root(service)/Path(project).name/snapshot['id']


def model_path(service, project, snapshot, *, touch=False):
    original=snapshot.get('model')
    if original:
        path = resolve_workspace_reference(service.workspace, original)
        if path.is_file(): return path
    path=cache_dir(service,project,snapshot)/'model.glb'
    if path.is_file():
        if touch: os.utime(path,None)
        return path
    return None


def ensure(service,project,snapshot,*,selected=False,background=False):
    path=model_path(service,project,snapshot,touch=True)
    if time.monotonic()-getattr(service,"history_last_prune",0)>10:
        prune(service,protected={path.parent} if path else set())
        service.history_last_prune=time.monotonic()
    if path:return {'status':'ready','path':str(path)}
    if not snapshot.get('source') or snapshot.get('status')=='incomplete':
        return {'status':'missing','error':'此记录尚无完整可预览模型'}
    key='history:'+str(project)+':'+snapshot['id']
    with service.lock:
        record=service.records.get(key)
        if not hasattr(service,'history_latest'): service.history_latest={}
        if not hasattr(service,'history_selected'): service.history_selected={}
        if not background:(service.history_selected if selected else service.history_latest)[str(project)]=snapshot['id']
        if record is None or record.get('status') in {'ready','deferred'}:
            record={'status':'queued','selected':selected,'background':background};service.records[key]=record
            service.jobs.put((0 if selected else 20 if background else 1,time.monotonic(),{'history':True,'key':key,'project':str(project),'snapshot':snapshot}))
        elif not background and (record.get('background') or selected and not record.get('selected')) and record['status']=='queued':
            record.update(selected=selected,background=False)
            service.jobs.put((0 if selected else 1,time.monotonic(),{'history':True,'key':key,'project':str(project),'snapshot':snapshot}))
        return dict(record)


def build(service, job):
    key,project,snapshot=job['key'],job['project'],job['snapshot']
    with service.lock:
        record=service.records[key]
        if record.get('status')!='queued':return
        targets=service.history_selected if record.get('selected') else service.history_latest
        if not record.get('background') and targets.get(str(project))!=snapshot['id']:
            record['status']='deferred'; return
        record['status']='building'
    output=cache_dir(service,project,snapshot)
    try:
        output.mkdir(parents=True,exist_ok=True)
        from .preview_service import resolve_cad_python
        from ..snapshot_paths import imported
        python=resolve_cad_python(None if imported(service.workspace) else snapshot.get('cad_python'))
        request={'workspace':str(service.workspace),'run':str(Path(project)/'runs'/snapshot['revision']),
                 'source':snapshot['source'],'source_bindings':snapshot.get('files',{}),'output':str(output),
                 'model_cache':str(output/'model.glb'),'traces':[],'input_version':snapshot['id']}
        write_json(output/'request.json',request)
        environment=dict(os.environ)
        # The CAD environment's own libraries must accompany its interpreter on Linux.
        lib=Path(python).resolve().parent.parent/'lib'
        if os.name!='nt' and lib.is_dir():environment['LD_LIBRARY_PATH']=str(lib)+os.pathsep+environment.get('LD_LIBRARY_PATH','')
        with (output/'build.log').open('w') as log:
            with service.lock:
                if service.closed.is_set():return
                service.process=subprocess.Popen([str(python),'-X','utf8','-m',
                    'simulation_ready_cad_workflow.validation_pipeline.dashboard.preview_worker',str(output/'request.json')],
                    cwd=Path(__file__).resolve().parents[3],env=environment,stdout=log,stderr=subprocess.STDOUT)
            try:code=service.process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                service.process.kill();service.process.wait(timeout=5)
                raise TimeoutError('预览转换超时，保存的模型和源码仍可访问')
        result=read_json(output/'result.json')
        if code or not (output/'model.glb').is_file():raise ValueError(result.get('error') or '预览转换失败，原始记录已保留')
        with service.lock:record.update(status='ready',error=None)
        prune(service,protected={output})
    except Exception as exc:
        with service.lock:record.update(status='failed',error=str(exc))
    finally:
        with service.lock:service.process=None


def usage(service):
    root=cache_root(service)
    return sum(f.stat().st_size for f in root.glob('*/*/*') if f.is_file()) if root.exists() else 0


def prune(service,protected=(),limit=CACHE_BYTES):
    root=cache_root(service)
    protected=set(protected)
    with service.lock:
        for paths in getattr(service,"image_inputs",{}).values():protected.update(Path(path).parent for path in paths)
    entries=[]
    for directory in root.glob('*/*'):
        if not directory.is_dir():continue
        files=[p for p in directory.iterdir() if p.is_file()]
        model=directory/'model.glb'
        access=max((f.stat().st_mtime for f in files if f.suffix.lower() in {'.glb','.png'}),default=directory.stat().st_mtime)
        entries.append((access,
                        sum(p.stat().st_size for p in files),directory))
    total=sum(size for _,size,_ in entries)
    for accessed,size,directory in sorted(entries):
        if total<=limit:break
        if directory in protected or time.time()-accessed<10:continue
        shutil.rmtree(directory);total-=size
    service.history_cache_bytes=total
    return total
