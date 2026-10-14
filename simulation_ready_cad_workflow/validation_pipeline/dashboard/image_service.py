"""One image worker separate from the GLB converter; both share the disposable cache."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

from ..trajectory.capture import read_json,write_json
from ..trajectory.store import HistoryStore,project_from_path
from . import history_preview
from .preview_service import _service,_slim_preview,_cache_root,_inputs,resolve_cad_python

PROFILE='workbench-png-1'


def model_reference(workspace,run,snapshot_id=None,*,prepare=False):
    workspace,run=Path(workspace).resolve(),Path(run).resolve();project=project_from_path(run)
    if project is None:raise ValueError('模型没有项目归属')
    store=HistoryStore(project);converter=_service(workspace)
    if snapshot_id:
        snapshot=store.snapshot(snapshot_id)
        if not snapshot:
            from .preview_service import preview_timeline
            legacy=next((item for item in preview_timeline(workspace_root=workspace,run_dir=run) if item['id']==snapshot_id),None)
            if legacy:snapshot={'id':snapshot_id,'revision':run.name,'model':str(_cache_root(workspace,run)/legacy['model'])}
        if not snapshot or snapshot['revision']!=run.name:raise ValueError('模型不属于所选版本')
        candidates=[snapshot]
    else:candidates=reversed(store.history(run.name)['items'])
    source_candidate=None
    for snapshot in candidates:
        if source_candidate is None and snapshot.get('source') and snapshot.get('status')!='incomplete':source_candidate=snapshot
        model=history_preview.model_path(converter,project,snapshot,touch=True)
        up=snapshot.get('up_axis')
        if not up:
            scene=read_json(snapshot.get('files',{}).get('scene.json',''))
            gravity=scene.get('gravity_m_s2')
            if isinstance(gravity,list) and len(gravity)==3 and all(isinstance(x,(float,int)) for x in gravity):
                magnitude=sum(x*x for x in gravity)**.5
                if magnitude:up=[-x/magnitude for x in gravity]
        if model:return {'identity':[str(run),snapshot['id']], 'model':str(model),'snapshot':snapshot['id'],
                         'up_axis':up or [0,1,0],'python':resolve_cad_python(snapshot.get('cad_python'))}
        if snapshot_id and prepare:
            progress=history_preview.ensure(converter,project,snapshot,selected=True)
            return {'status':progress['status'],'error':progress.get('error'),'identity':[str(run),snapshot_id]}
    if snapshot_id:return {'status':'missing','error':'该记录没有可用的模型'}
    preview=_slim_preview(run)
    if not preview:
        candidate=read_json(_cache_root(workspace,run)/'preview.json')
        _,inputs,_=_inputs(workspace,run)
        if candidate.get('inputs')==[list(x) for x in inputs]:preview=candidate
    raw=preview.get('model_path')
    if raw and Path(raw).is_file():
        stat=Path(raw).stat()
        return {'identity':[str(run),preview.get('input_version'),stat.st_size,stat.st_mtime_ns],
                'model':raw,'up_axis':preview.get('up_axis') or [0,1,0],'python':resolve_cad_python()}
    if prepare and source_candidate:
        return history_preview.ensure(converter,project,source_candidate,background=True)
    return {'status':'missing','error':'项目尚无可用的原始模型'}


class ImageService:
    def __init__(self,workspace):
        self.workspace=Path(workspace).resolve();self.lock=threading.Lock();self.jobs=queue.PriorityQueue()
        self.records={};self.by_id={};self.latest_cover={};self.closed=threading.Event();self.process=None
        for path in history_preview.cache_root(_service(workspace)).glob('*/image-*/request.json'):
            request=read_json(path);result=read_json(path.parent/'result.json')
            if request.get('key') and result.get('status')=='ready' and all(Path(p).is_file() for p in result.get('images',[])):
                value={**result,'id':path.parent.name,'key':request['key'],'project':request.get('project')}
                self.records[request['key']]=value;self.by_id[value['id']]=value
        self.thread=threading.Thread(target=self._loop,name='workbench-images',daemon=True);self.thread.start()

    def request(self,project,references,*,kind='cover',view='iso'):
        if view not in {'iso','front','side','top'}:raise ValueError('Unknown camera view')
        for reference in references:
            if not reference.get('model'):return {'status':reference.get('status','missing'),'error':reference.get('error')}
        key=json.dumps([PROFILE,kind,view,[r['identity'] for r in references]],sort_keys=True)
        with self.lock:
            current=self.records.get(key)
            if current and current.get('status')=='ready' and not all(Path(p).is_file() for p in current['images']):current=None
            if current and current.get('status')!='obsolete':
                for raw in current.get('images',[]):
                    if Path(raw).is_file():os.utime(raw,None)
                return dict(current)
            identity='image-'+uuid.uuid4().hex[:16]
            output=history_preview.cache_root(_service(self.workspace))/Path(project).name/identity
            request={'key':key,'project':str(project),'models':[r['model'] for r in references],
                     'up_axis':references[0].get('up_axis'),'python':references[0].get('python'),
                     'width':240 if kind=='cover' else 960,'height':160 if kind=='cover' else 640,
                     'view':view,'output':str(output),'kind':kind}
            result={'id':identity,'key':key,'project':str(project),'status':'queued'}
            self.records[key]=result;self.by_id[identity]=result
            converter=_service(self.workspace)
            with converter.lock:converter.image_inputs[identity]=request['models']
            if kind=='cover':self.latest_cover[str(project)]=key
            self.jobs.put((0 if kind=='compare' else 10,time.monotonic(),request))
            return dict(result)

    def _loop(self):
        while not self.closed.is_set():
            _,_,request=self.jobs.get()
            if request is None:return
            record=self.records[request['key']]
            if request['kind']=='cover' and self.latest_cover.get(request['project'])!=request['key']:
                record['status']='obsolete'
                converter=_service(self.workspace)
                with converter.lock:converter.image_inputs.pop(record['id'],None)
                continue
            output=Path(request['output'])
            try:
                output.mkdir(parents=True,exist_ok=True);write_json(output/'request.json',request)
                record['status']='building'
                from ..snapshot_paths import imported
                python=resolve_cad_python(None if imported(self.workspace) else request.get('python'));environment=dict(os.environ)
                library=Path(python).resolve().parent.parent/'lib'
                if os.name!='nt' and library.is_dir():environment['LD_LIBRARY_PATH']=str(library)+os.pathsep+environment.get('LD_LIBRARY_PATH','')
                with (output/'build.log').open('w') as log:
                    with self.lock:
                        if self.closed.is_set():return
                        self.process=subprocess.Popen([str(python),'-m','simulation_ready_cad_workflow.validation_pipeline.dashboard.image_worker',str(output/'request.json')],
                        cwd=Path(__file__).resolve().parents[3],env=environment,stdout=log,stderr=subprocess.STDOUT)
                    try:code=self.process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        self.process.kill();self.process.wait(timeout=5);raise TimeoutError('图片生成超时')
                result=read_json(output/'result.json')
                if code or result.get('status')!='ready':raise ValueError(result.get('error') or '图片生成失败')
                with self.lock:record.update(result)
                history_preview.prune(_service(self.workspace),protected={output})
            except Exception as exc:record.update(status='failed',error=str(exc))
            finally:
                self.process=None
                converter=_service(self.workspace)
                with converter.lock:converter.image_inputs.pop(record['id'],None)

    def read_image(self,identity,index):
        record=self.by_id.get(identity)
        if not record or record.get('status')!='ready':raise ValueError('图片尚未就绪')
        files=record.get('images',[])
        if index<0 or index>=len(files):raise ValueError('Unknown image')
        path=Path(files[index]);os.utime(path,None)
        return path

    def close(self):
        with self.lock:
            self.closed.set();process=self.process
        self.jobs.put((-1,time.monotonic(),None))
        if process and process.poll() is None:process.terminate()
        self.thread.join(timeout=3)
