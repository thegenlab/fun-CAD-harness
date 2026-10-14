"""Executed source/STEP delivery evidence, without printing requirements."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from ..runtime.bounded_subprocess import run_bounded_subprocess
from .static_delivery import binding, check_binding, digest, read, write

VERIFIER_ID='step_delivery_v1'
REGISTRATION_FILE='step-delivery-registration.json'


def implementation_binding():
    from . import asset, static_delivery_worker
    from ..runtime import bounded_subprocess
    return binding([__file__,Path(__file__).with_name('step_delivery_worker.py'),
                    asset.__file__,static_delivery_worker.__file__,bounded_subprocess.__file__])


def validate_request(project,design,path):
    project,design,path=map(lambda p:Path(p).resolve(),(project,design,path))
    if path.is_relative_to(design):raise ValueError('The verifier request belongs outside the author design')
    request=read(path)
    from ..snapshot_paths import resolve, binding_paths
    workspace = project.parents[2]
    if request['verifier_id']!=VERIFIER_ID or resolve(workspace, request['scene_path']) != design/'scene.json':
        raise ValueError('Wrong STEP verifier or scene')
    criteria=read(project/'criteria.json');lock=read(project/'criteria.lock.json')
    if (criteria['task_text']!=(project/'task.md').read_text(encoding='utf-8')
            or lock['criteria_sha256']!=digest(project/'criteria.json')
            or lock['task_sha256']!=digest(project/'task.md')):raise ValueError('Task or criteria binding changed')
    declarations=request['criteria']
    if not declarations or len({r['criterion']['id'] for r in declarations})!=len(declarations):
        raise ValueError('Bind unique frozen delivery criteria')
    for row in declarations:
        if row['criterion'] not in criteria['criteria'] or row['coverage_mode'] not in ('static_only','supplemental'):
            raise ValueError('Invalid delivery criterion or coverage mode')
    scene=read(design/'scene.json')
    required={'scene.json','parts.json',request['model_source'],request['assembly_step'],
              *scene.get('authoring_sources',[]),*(b['source'] for b in scene['bodies']),*request['documents']}
    if set(request['source_files'])!=required:raise ValueError('Incomplete STEP source snapshot')
    for name in required:
        if Path(name).is_absolute() or not (design/name).resolve().is_relative_to(design):
            raise ValueError('Keep source paths inside the current design')
    sources=binding(design/name for name in required)
    if sources!=binding_paths(workspace, request['source_bindings']):raise ValueError('Delivered source changed')
    return request,sources,binding(project/name for name in ('task.md','criteria.json','criteria.lock.json'))


def execute(project,design,request_path,out,cad_python):
    project,design,request_path,out=map(lambda p:Path(p).resolve(),(project,design,request_path,out))
    request,sources,criteria=validate_request(project,design,request_path)
    if out.exists() or out.is_relative_to(design):raise ValueError('Use new program-owned execution evidence')
    out.mkdir(parents=True)
    receipt={'verifier_id':VERIFIER_ID,'scene_path':str(design/'scene.json'),'project_path':str(project),
             'request_path':str(request_path),'request_sha256':digest(request_path),'source_bindings':sources,
             'criteria_bindings':criteria,'implementation_bindings':implementation_binding(),
             'started_at':datetime.now(timezone.utc).isoformat(),'outcome':'incomplete',
             'scope':'Executed source rebuild and STEP/scene/physical-inventory correspondence; documentation content and assembly completeness remain independently reviewed. No physical or print acceptance.'}
    started=time.monotonic();inspection=out/'inspection.json';payload=out/'inspection-request.json'
    write(payload,{'design':str(design),'request':request,'output':str(inspection),'rebuilt_design':str(out/'rebuilt_design')})
    try:
        from .execution import run_compute
        workspace=project.parents[2]
        result=run_compute([str(cad_python),'-X','utf8','-m',
            'simulation_ready_cad_workflow.validation_pipeline.slim_harness.step_delivery_worker',str(payload)],
            cwd=workspace,workspace=workspace,project=project,inputs=[design,payload],outputs=[out],
            timeout_seconds=1800,max_rss_bytes="auto",runner=run_bounded_subprocess)
        stdout,stderr=out/'rebuild.stdout.log',out/'rebuild.stderr.log'
        stdout.write_bytes(result.stdout);stderr.write_bytes(result.stderr)
        receipt['execution']={'returncode':result.returncode,'logs':binding([stdout,stderr])}
        if result.returncode:raise ValueError('Source/STEP rebuild failed; see '+str(stderr))
        checked=read(inspection)
        if checked.get('completed') is not True:raise ValueError('STEP inspection incomplete')
        if not all(check_binding(x) for x in (sources,criteria,receipt['implementation_bindings'],checked['runtime_bindings'])):
            raise ValueError('Source, task or runtime changed during verification')
        receipt.update(outcome='completed',inspection_path=str(inspection),runtime_bindings=checked['runtime_bindings'],
                       artifact_bindings=binding([payload,inspection,*checked['rebuilt_files']]))
    except Exception as exc:
        receipt.update(outcome='error',error=f'{type(exc).__name__}: {exc}')
    receipt['wall_seconds']=time.monotonic()-started
    receipt['receipt_id']=VERIFIER_ID+':'+digest(request_path)[:16]+':'+out.name
    write(out/'receipt.json',receipt)
    return receipt


def _load_entry(project,scene_path,entry):
    from .print_delivery import PrintStaticEvidence,canonical
    project,scene_path=Path(project).resolve(),Path(scene_path).resolve()
    from ..snapshot_paths import resolve, binding_paths, implementation_matches
    workspace = project.parents[2]
    mapped = lambda value: resolve(workspace, value)
    bindings = lambda value: binding_paths(workspace, value)
    path=mapped(entry['receipt_path'])
    if path.is_relative_to(scene_path.parent) or digest(path)!=entry['receipt_sha256']:
        raise ValueError('Receipt location or bytes changed')
    request,sources,criteria=validate_request(project,scene_path.parent,mapped(entry['request_path']))
    receipt=read(path)
    if (receipt['verifier_id']!=VERIFIER_ID or receipt['outcome']!='completed'
            or mapped(receipt['scene_path'])!=scene_path or mapped(receipt['project_path'])!=project
            or bindings(receipt['source_bindings'])!=sources or bindings(receipt['criteria_bindings'])!=criteria
            or receipt['request_sha256']!=digest(mapped(entry['request_path']))
            or not implementation_matches(workspace, receipt['implementation_bindings'], implementation_binding())
            or receipt['execution']['returncode']!=0):raise ValueError('Stale or incomplete STEP receipt')
    if not all(check_binding(bindings(receipt[k])) for k in ('runtime_bindings','artifact_bindings')) or not check_binding(bindings(receipt['execution']['logs'])):
        raise ValueError('Executed STEP evidence changed')
    checked=read(mapped(receipt['inspection_path']))
    if checked.get('completed') is not True:raise ValueError('No completed source rebuild')
    return tuple(PrintStaticEvidence(receipt['receipt_id']+':'+row['criterion']['id'],str(path),entry['receipt_sha256'],
        str(scene_path),digest(project/'criteria.json'),(str(path),str(mapped(receipt['inspection_path'])),
        *(str(scene_path.parent/name) for name in request['documents'])),canonical(row['criterion']),
        row['coverage_mode'],receipt['scope']) for row in request['criteria'])


def load_registered_step_evidence(project,scene_path):
    path=Path(project)/REGISTRATION_FILE
    if not path.is_file():return (),[]
    try:return _load_entry(project,scene_path,read(path)),[]
    except (OSError,ValueError,KeyError,TypeError) as exc:return (),['STEP receipt rejected: '+str(exc)]


def ensure_step_delivery(project,scene_path,attempt,requests,*,cad_python=sys.executable):
    """Execute an explicit program-owned STEP evidence request."""
    project,scene_path,attempt=map(lambda p:Path(p).resolve(),(project,scene_path,attempt))
    criteria={c['id']:c for c in read(project/'criteria.json')['criteria']}
    declarations=[]
    for row in requests:
        target=row.get('target','')
        if row.get('action')!='local_measurement' or target not in ('step_delivery:static_only','step_delivery:supplemental'):continue
        declarations.append({'criterion':criteria[row['criterion_id']],'coverage_mode':target.split(':')[1]})
    if not declarations:raise ValueError('No explicit STEP delivery evidence request')
    existing,_=load_registered_step_evidence(project,scene_path)
    if all(any(e.criterion_json==json.dumps(d['criterion'],sort_keys=True,ensure_ascii=False,separators=(',',':'))
               and e.coverage_mode==d['coverage_mode'] for e in existing) for d in declarations):
        return {'status':'ready','receipt_ids':[e.receipt_id for e in existing],'evidence_paths':list(existing[0].evidence_paths)}
    declared_ids={row['criterion']['id'] for row in declarations}
    for evidence in existing:
        criterion=json.loads(evidence.criterion_json)
        if criterion['id'] not in declared_ids:
            declarations.append({'criterion':criterion,'coverage_mode':evidence.coverage_mode})
    design=scene_path.parent;scene=read(scene_path);sources=scene.get('authoring_sources',[])
    body_sources={b['source'] for b in scene['bodies']}
    assemblies=[s for s in sources if Path(s).suffix.lower() in ('.step','.stp') and s not in body_sources]
    if not sources or Path(sources[0]).suffix!='.py' or len(assemblies)!=1:
        return {'status':'needs_revision','issues':['Declare the Asset build source first in authoring_sources and one complete assembly STEP dependency; the verifier does not guess a build recipe.'],'evidence_paths':[]}
    documents=[s for s in sources if Path(s).suffix.lower() in ('.md','.html','.htm','.txt')]
    if not documents:return {'status':'needs_revision','issues':['No declared delivery documentation.'],'evidence_paths':[]}
    files=sorted({'scene.json','parts.json',*sources,*body_sources})
    request={'verifier_id':VERIFIER_ID,'scene_path':str(scene_path),'model_source':sources[0],
             'assembly_step':assemblies[0],'documents':documents,'source_files':files,
             'source_bindings':binding(design/name for name in files),'criteria':declarations}
    counter=len(list(attempt.glob('step-delivery-*')))+1
    request_path=attempt/f'step-delivery-request-{counter:03d}.json';write(request_path,request)
    out=attempt/f'step-delivery-{counter:03d}'
    receipt=execute(project,design,request_path,out,cad_python)
    receipt_path=out/'receipt.json'
    if receipt['outcome']!='completed':return {'status':'needs_revision','issues':[receipt['error']],'evidence_paths':[str(receipt_path)]}
    entry={'verifier_id':VERIFIER_ID,'request_path':str(request_path),'receipt_path':str(receipt_path),'receipt_sha256':digest(receipt_path)}
    evidence=_load_entry(project,scene_path,entry)
    write(project/REGISTRATION_FILE,entry)
    return {'status':'ready','receipt_ids':[e.receipt_id for e in evidence],'evidence_paths':[str(receipt_path),receipt['inspection_path']]}


def refresh_registered_step_delivery(project,scene_path,attempt,*,cad_python=sys.executable):
    """Rebuild stale registered delivery evidence against the current revision."""
    project,scene_path,attempt=map(Path,(project,scene_path,attempt))
    registration=project/REGISTRATION_FILE
    if not registration.is_file():return None
    _,errors=load_registered_step_evidence(project,scene_path)
    if not errors:return None
    try:
        entry=read(registration);previous=read(entry['request_path'])
        receipt=read(entry['receipt_path'])
        if (entry.get('verifier_id')!=VERIFIER_ID or previous.get('verifier_id')!=VERIFIER_ID
                or digest(entry['receipt_path'])!=entry['receipt_sha256']
                or digest(entry['request_path'])!=receipt['request_sha256']):
            raise ValueError('Prior STEP registration bytes changed')
        criteria=read(project/'criteria.json')['criteria']
        requests=[]
        for row in previous['criteria']:
            if row['criterion'] not in criteria or row['coverage_mode'] not in ('static_only','supplemental'):
                raise ValueError('Registered STEP criteria changed; retain the original declaration instead of guessing by ID')
            requests.append({'criterion_id':row['criterion']['id'],'action':'local_measurement',
                             'target':'step_delivery:'+row['coverage_mode']})
        attempt.mkdir(parents=True,exist_ok=True)
        write(attempt/'step-registration.before.json',entry)
        return ensure_step_delivery(project,scene_path,attempt,requests,cad_python=cad_python)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {'status':'needs_revision','issues':['STEP registration refresh failed: '+str(exc)],'evidence_paths':[]}
