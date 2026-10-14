"""Capture while workers run, even when no dashboard is connected."""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

from .store import HistoryStore, project_from_path, revision_from_path

_CURRENT = contextvars.ContextVar('author_capture', default=None)
_MODEL = {'.step','.stp','.glb','.obj','.stl'}
_TEXT = {'.py','.json','.yaml','.yml','.toml','.txt','.scad'}


def read_json(path):
    try:
        data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return data if isinstance(data,dict) else {}
    except (OSError,ValueError):
        return {}


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex[:8]+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');tmp.replace(path)


def now_ms():
    return int(time.time()*1000)


def snapshot_attempt(workspace, attempt, result=None, *, recovered=False):
    workspace,attempt=Path(workspace).resolve(),Path(attempt).resolve()
    result=result or read_json(attempt/'result.json')
    meta=read_json(attempt/'capture.json')
    from ..dashboard.path_resolution import resolve_workspace_reference
    source=meta.get('source') or result.get('source','')
    if source:source=str(resolve_workspace_reference(workspace,source))
    project=project_from_path(source) if source else None
    if meta.get('project'):
        project=resolve_workspace_reference(workspace,meta['project'])
    if not project or project.parent != workspace/'.runtime_runs/projects' or not (project/'workflow.json').is_file():
        return None
    revision=meta.get('revision') or revision_from_path(source)
    if not revision:
        return None
    design=attempt/'design'
    files={str(f.relative_to(design)):str(f) for f in design.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    # Template preparation retains the compiled model and geometry manifest here;
    # its live canonical source directory is not an immutable source snapshot.
    for name in ('prepared.json','geometry/geometry.json'):
        if (attempt/name).is_file():files.setdefault(name,str(attempt/name))
    source_name=Path(source).name
    if (attempt/source_name).is_file():
        files.setdefault(source_name,str(attempt/source_name))
    geometry=read_json(attempt/'geometry/geometry.json')
    model=geometry.get('model_file') or result.get('model')
    if model:model=str(resolve_workspace_reference(workspace,model))
    if not model or not Path(model).is_file():model=None
    scene=read_json(design/'scene.json')
    gravity=scene.get('gravity_m_s2') or [0,-9.81,0]
    if not isinstance(gravity,list) or len(gravity)!=3 or not all(isinstance(x,(int,float)) for x in gravity):
        gravity=[0,-9.81,0]
    magnitude=sum(float(x)**2 for x in gravity)**.5
    from .configuration import model_binding
    value={'model_binding':model_binding(geometry),'id':meta.get('snapshot_id') or attempt.name,'revision':revision,'turn':meta.get('turn'),
           'at_ms':meta.get('geometry_ready_ms') or meta.get('completed_ms') or meta.get('started_ms'),
           'started_ms':meta.get('started_ms'),'geometry_ready_ms':meta.get('geometry_ready_ms'),'completed_ms':meta.get('completed_ms'),
           'capture':'recovered' if recovered and not meta else 'build',
           'status':result.get('status') or meta.get('status','building'),
           'model':model,'source':str(design/'scene.json') if (design/'scene.json').is_file() else None,
           'files':files,'attempt':str(attempt),'command':result.get('command') or meta.get('command'),
           'source_complete':bool(scene and files),'cad_python':meta.get('cad_python'),
           'up_axis':[-float(x)/magnitude for x in gravity] if magnitude else [0,1,0],
           'error':result.get('issues') or geometry.get('errors') or [],
           'geometry_ready':bool(model),'test_ids':[x.get('test_id') or x.get('id') for x in result.get('tests',[]) if isinstance(x,dict)],'source_scene_sha256':geometry.get('scene_sha256') or result.get('source_scene_sha256'),'config_path':meta.get('config_path')}
    HistoryStore(project).put_snapshot(value)
    return value


def _bound_revision(project, path):
    revision=revision_from_path(path or '')
    if revision:return revision
    if os.environ.get('SIMREADY_TRAJECTORY_PROJECT')==str(project):
        revision=os.environ.get('SIMREADY_TRAJECTORY_REVISION')
        if revision:return revision
    revision=read_json(Path(project)/'workflow.json').get('revision')
    return f'r{revision:02d}' if isinstance(revision,int) and not isinstance(revision,bool) and revision>0 else ''


_PUBLISHER_PROCESSES=[]


def launch_publisher(workspace, mode, path):
    """Launch outside the author process; publication never waits for indexing."""
    environment=dict(os.environ)
    environment.pop('SIMREADY_CONSTRUCTION_PROJECT',None)
    environment.pop('SIMREADY_CONSTRUCTION_GUARDED_REQUEST',None)
    environment['PYTHONPATH']=str(Path(__file__).resolve().parents[3])+os.pathsep+environment.get('PYTHONPATH','')
    with (Path(path).parent/'publisher.log').open('ab') as log:
        process=subprocess.Popen([os.environ.get('SIMREADY_CAD_PYTHON') or sys.executable,'-X','utf8','-m',
            'simulation_ready_cad_workflow.validation_pipeline.trajectory.publisher',mode,str(path)],
            cwd=workspace,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=os.name!='nt',
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0)
    _PUBLISHER_PROCESSES[:]=[p for p in _PUBLISHER_PROCESSES if p.poll() is None]
    _PUBLISHER_PROCESSES.append(process)
    return process


_ATTEMPT_PUBLISHERS={}


def _publish_attempt(workspace, attempt, result, *, project=None, destination=None, stage='started'):
    """Publish existing attempt files; never copy or rewrite validation evidence."""
    attempt=Path(attempt)
    meta=read_json(attempt/'capture.json')
    project=Path(meta['project']) if meta.get('project') else Path(project) if project else project_from_path(result.get('source',''))
    if project is None:return
    revision=meta.get('revision') or _bound_revision(project,destination or result.get('source',''))
    turn=meta.get('turn') or os.environ.get('SIMREADY_TRAJECTORY_TURN')
    if not turn:
        turn=read_json(project/'workflow.json').get('active_turn',{}).get('id')
    meta.update(project=str(project),revision=revision,turn=turn,source=meta.get('source') or result.get('source'),
                command=result.get('command'),status=result.get('status') or stage,
                cad_python=os.environ.get('SIMREADY_CAD_PYTHON') or sys.executable)
    meta.update(workspace=str(Path(workspace).resolve()),publish_result=dict(result),publication_stage=stage)
    # Template attempts have repeated leaf names (e.g. r01/.../a001/prepared).
    # Include their retained project-relative location without changing author-tool IDs.
    identity='project-'+attempt.resolve().relative_to(project.resolve()).as_posix().replace('/','--') if attempt.resolve().is_relative_to(project.resolve()) else attempt.name
    meta.setdefault('snapshot_id',identity)
    config_path=os.environ.get('SIMREADY_TRAJECTORY_CONFIG')
    if config_path:meta['config_path']=config_path
    meta.setdefault('started_ms',now_ms())
    meta.setdefault('owner_pid',os.getpid())
    if stage=='geometry_ready':meta.setdefault('geometry_ready_ms',now_ms())
    if stage=='finished':meta.setdefault('completed_ms',now_ms())
    write_json(attempt/'capture.json',meta)
    # The parent turn publisher is a sibling of the bounded author process.
    # Its durable queue survives command completion and process-tree cleanup.
    queue=project/'trajectory/pending'/f'{meta["snapshot_id"]}.json'
    write_json(queue,{'attempt':str(attempt),'workspace':str(Path(workspace).resolve())})
    control=read_json(os.environ.get('SIMREADY_TRAJECTORY_PUBLISHER',''))
    if control.get('status')=='recording' and control.get('project')==str(project):return
    previous=_ATTEMPT_PUBLISHERS.get(str(attempt))
    if previous is None or previous.poll() is not None:
        _ATTEMPT_PUBLISHERS[str(attempt)]=launch_publisher(workspace,'attempt',attempt/'capture.json')


def publish_attempt(workspace, attempt, result, **kwargs):
    try:
        _publish_attempt(workspace,attempt,result,**kwargs)
    except Exception as exc:
        # Recording is observational: a database error cannot change a physical verdict.
        import logging
        result['trajectory_error']=str(exc)
        logging.getLogger(__name__).warning('History recording failed for %s: %s',attempt,exc)
        try:write_json(Path(attempt)/'capture-error.json',{'error':str(exc),'at_ms':now_ms()})
        except OSError:pass


def _same_file(left,right):
    if not right.is_file() or left.stat().st_size!=right.stat().st_size:return False
    with left.open('rb') as a,right.open('rb') as b:
        while True:
            block=a.read(256*1024)
            if block!=b.read(256*1024):return False
            if not block:return True


class FileObserver:
    def __init__(self, store, design, turn):
        self.store,self.design,self.turn=store,Path(design).resolve(),turn
        self.revision=revision_from_path(design)
        self.seen=None;self.since=0.;self.files={}
        prior=store.history(self.revision,limit=1)['items']
        if prior:self.files=prior[-1].get('files',{})
        self.lock=threading.Lock()
        self.last_capture=None;self.committed_stamps=None

    def paths(self):
        result={}
        if not self.design.is_dir():return result
        for directory, folders, files in os.walk(self.design):
            folders[:]=[name for name in folders if name not in {'__pycache__','.git','.presim_gate','.runtime_runs'}]
            for name in files:
                f=Path(directory)/name
                if f.suffix.lower() in _TEXT or (f.parent==self.design and f.suffix.lower() in _MODEL):
                    result[f.relative_to(self.design).as_posix()]=f
        scene=read_json(self.design/'scene.json')
        refs=list(scene.get('authoring_sources')) if isinstance(scene.get('authoring_sources'),list) else []
        bodies=scene.get('bodies') if isinstance(scene.get('bodies'),list) else []
        refs += [b.get('source') for b in bodies if isinstance(b,dict)]
        for name in refs:
            if isinstance(name,str):
                p=(self.design/name).resolve()
                if p.is_relative_to(self.design) and p.is_file():result[name]=p
        return result

    def tick(self, force=False):
        with self.lock:
            paths=self.paths()
            stamps={name:(p.stat().st_size,p.stat().st_mtime_ns) for name,p in paths.items()}
            if stamps==self.committed_stamps:return
            now=time.monotonic()
            if stamps!=self.seen:
                self.seen=stamps;self.since=now
                if not force:return
            if not force and now-self.since<1.:return
            if not paths:return
            for known in self.store.recent_builds(self.revision):
                bindings=known.get('files',{})
                if known.get('model') and set(bindings)==set(paths) and all(_same_file(path,Path(bindings[name])) for name,path in paths.items()):
                    # The normal build already retained these exact inputs and its GLB.
                    self.files=dict(bindings);self.committed_stamps=stamps
                    return
            changed={name:p for name,p in paths.items() if name not in self.files or not _same_file(p,Path(self.files[name]))}
            removed=set(self.files)-set(paths)
            if not changed and not removed:
                self.committed_stamps=stamps
                return
            identity='observed-'+uuid.uuid4().hex[:16]
            folder=self.store.root/'files'/identity
            bindings={name:ref for name,ref in self.files.items() if name in paths}
            for name,path in changed.items():
                target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                if (path.stat().st_size,path.stat().st_mtime_ns)!=stamps[name]:
                    self.seen=None
                    return
                bindings[name]=str(target)
            # Keep a self-contained scene directory through copies of the changed files and
            # immutable references. Conversion resolves references from the manifest.
            self.files=bindings
            scene=read_json(Path(bindings['scene.json'])) if 'scene.json' in bindings else {}
            source=bindings.get('scene.json')
            if not scene:
                source=next((bindings[n] for n in ('assembly.step','assembly.stp','assembly.glb','assembly.obj','assembly.stl') if n in bindings),None)
            if not source:
                source=next((ref for name,ref in bindings.items() if Path(name).suffix.lower() in _MODEL),None)
            complete=bool(source)
            if scene:
                bodies=scene.get('bodies')
                complete=isinstance(bodies,list) and bool(bodies) and all(isinstance(b,dict) and isinstance(b.get('source'),str) and b['source'] in bindings for b in bodies)
            model=source if source and Path(source).suffix.lower()=='.glb' else None
            if model:
                with Path(model).open('rb') as stream: header=stream.read(12)
                valid=(len(header)==12 and header[:4]==b'glTF' and int.from_bytes(header[8:12],'little')==Path(model).stat().st_size)
                if not valid:model=None;complete=False
            stamp=now_ms()
            value={'id':identity,'revision':self.revision,'turn':self.turn,'at_ms':stamp,
                   'capture':'observed','status':'observed' if complete else 'incomplete','model':model,
                   'source':source,'files':bindings,'source_complete':False,'geometry_ready':bool(model),
                   'observation_note':'文件稳定后观测；源码与几何的精确生成关系未确认。',
                   'cad_python':os.environ.get('SIMREADY_CAD_PYTHON'),'config_path':getattr(self,'config_path',None),'changed_files':sorted(changed),'removed_files':sorted(removed)}
            write_json(self.store.root/'snapshots'/(identity+'.json'),value)
            self.store.put_snapshot(value);self.last_capture=stamp;self.committed_stamps=stamps


class TurnCapture:
    def __init__(self, workspace, output_dir, readonly=False, runtime_config=None):
        self.design=Path(workspace).resolve();self.output=Path(output_dir).resolve()
        self.project=project_from_path(output_dir)
        self.turn=self.output.name
        if self.project:
            try:self.turn=self.output.relative_to(self.project/"worker_turns").parts[0]
            except ValueError:pass
        self.store=HistoryStore(self.project) if self.project and (self.project/'workflow.json').is_file() else None
        self.revision=_bound_revision(self.project,self.design) if self.project else ''
        self.readonly=readonly
        self.observer=None
        self.error=None
        self.cpu_seconds=0.
        self.runtime_config=runtime_config
        self.config_path=self.output/("runtime_config."+uuid.uuid4().hex[:12]+".json") if runtime_config else None
        self.control_path=self.output/('publication.'+uuid.uuid4().hex[:12]+'.json')
        self.process=None

    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=True)
        if self.config_path:write_json(self.config_path,self.runtime_config)
        log=self.output/'events.jsonl'
        if log.is_file() and log.stat().st_size:
            target=self.output/('events.'+uuid.uuid4().hex[:12]+'.jsonl')
            log.replace(target)
            if self.store and self.store.path.is_file():self.store.archive_log(log,target)
        if self.store:
            write_json(self.output/'capture.json',{'started_ms':now_ms(),'turn':self.turn,'status':'recording'})
            write_json(self.control_path,{'workspace':str(self.project.parents[2]),'project':str(self.project),
                'design':str(self.design),'output':str(self.output),'turn':self.turn,'revision':self.revision,
                'readonly':self.readonly,'config_path':str(self.config_path) if self.config_path else None,
                'status':'recording','started_ms':now_ms(),'owner_pid':os.getpid()})
            try:self.process=launch_publisher(self.project.parents[2],'turn',self.control_path)
            except OSError as exc:
                self.error=str(exc)
                write_json(self.output/'capture-error.json',{'error':self.error,'at_ms':now_ms()})
        return self

    def _tick(self, final=False):
        if not self.store:return
        cpu_started=time.thread_time()
        completed=False
        for log in sorted(self.output.glob('events*.jsonl')):
            self.store.index_log(log,self.turn,observed=True,drain=final)
            completed=completed or self.store.tool_completed
        if self.observer:self.observer.tick(force=final or completed)
        self.store.set_state('capture',{'status':'recording','turn':self.turn,'updated_ms':now_ms(),
                                      'mode':'build_and_observed','error':self.error,'cpu_seconds':self.cpu_seconds})
        self.cpu_seconds+=time.thread_time()-cpu_started

    def __exit__(self, kind, error, traceback):
        if self.store:
            control=read_json(self.control_path)
            control.update(status='interrupted' if error else 'finished',ended_ms=now_ms())
            write_json(self.control_path,control)
            write_json(self.output/'capture.json',{'turn':self.turn,'ended_ms':now_ms(),'status':'interrupted' if error else 'finished','error':self.error})


class HarnessProgramCapture(TurnCapture):
    """Actual Harness commands in the existing public trajectory, without model usage."""
    def __init__(self, workspace, output_dir):
        super().__init__(workspace,output_dir,readonly=True)

    def __enter__(self):
        super().__enter__()
        write_json(self.output/'program.json',{'schema_version':'harness_program_capture_v1','source':'harness_program'})
        return self

    @contextmanager
    def command(self, command):
        """Caller fills aggregated_output and exit_code from the real subprocess result."""
        item={'id':'harness-'+uuid.uuid4().hex,'type':'command_execution',
              'command':'Harness program: '+str(command),'status':'running'}
        _append_public_event(self,{'type':'item.started','source':'harness_program','item':item})
        try:
            yield item
        except BaseException as exc:
            item['status']='failed'
            output=item.get('aggregated_output','')
            item['aggregated_output']=str(output)+('\n' if output else '')+f'{type(exc).__name__}: {exc}'
            raise
        else:
            item['status']='completed' if item.get('exit_code',0)==0 else 'failed'
        finally:
            _append_public_event(self,{'type':'item.completed','source':'harness_program','item':item})


def record_turn(method):
    @wraps(method)
    def recorded(self, workspace, prompt, output_dir, timeout_seconds, *args, **kwargs):
        from .configuration import worker_configuration
        config=worker_configuration(self,project_from_path(output_dir),workspace,kwargs.get('readonly',False))
        config['worker']['turn_timeout_seconds']=timeout_seconds
        with TurnCapture(workspace,output_dir,readonly=kwargs.get('readonly',False),runtime_config=config) as capture:
            token=_CURRENT.set(capture)
            try:return method(self,workspace,prompt,output_dir,timeout_seconds,*args,**kwargs)
            finally:_CURRENT.reset(token)
    return recorded


def turn_environment():
    env=dict(os.environ);capture=_CURRENT.get()
    if capture and capture.store:
        env['SIMREADY_TRAJECTORY_TURN']=capture.turn
        env['SIMREADY_TRAJECTORY_PROJECT']=str(capture.project)
        env['SIMREADY_TRAJECTORY_REVISION']=capture.revision
        if capture.process:env['SIMREADY_TRAJECTORY_PUBLISHER']=str(capture.control_path)
        if capture.config_path:env['SIMREADY_TRAJECTORY_CONFIG']=str(capture.config_path)
    return env


def _append_public_event(capture, event):
    if capture is None:return
    event=dict(event);event.setdefault('timestamp_ms',now_ms())
    with (capture.output/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event,ensure_ascii=False)+'\n')


def emit_public_event(event):
    _append_public_event(_CURRENT.get(),event)


class HistoryIndexer:
    """One background legacy import; live records are owned by workers/tools."""
    def __init__(self,workspace):
        self.workspace=Path(workspace).resolve();self.closed=threading.Event()
        self.thread=threading.Thread(target=self._run,name='history-import',daemon=True)
        self.error=None;self.ready=False
        self.thread.start()

    def _run(self):
        try:
            for project in (self.workspace/'.runtime_runs/projects').glob('*'):
                if self.closed.is_set():return
                if not (project/'workflow.json').is_file():continue
                store=HistoryStore(project)
                for log in sorted((project/'worker_turns').rglob('events*.jsonl')):
                    store.index_log(log,log.relative_to(project/'worker_turns').parts[0],drain=True)
                for manifest in (store.root/'snapshots').glob('*.json'):
                    value=read_json(manifest)
                    if value:store.put_snapshot(value)
            for attempt in (self.workspace/'.runtime_runs/author_tools').glob('*'):
                if self.closed.is_set():return
                if (attempt/'result.json').is_file():snapshot_attempt(self.workspace,attempt,recovered=True)
            self.ready=True
        except Exception as exc:self.error=str(exc)

    def close(self):
        self.closed.set();self.thread.join(timeout=5)
