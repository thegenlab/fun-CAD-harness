"""Read settings bound to the selected revision/snapshot, never today's defaults."""
from pathlib import Path
import re

from .path_resolution import resolve_workspace_reference
from ..trajectory.capture import read_json
from ..trajectory.store import HistoryStore, project_from_path


def public_settings(value):
    if isinstance(value,dict):
        return {key:('[redacted]' if re.sub(r'[_-]','',key).lower() in {'apikey','password','authorization','accesstoken','refreshtoken','secret','clientsecret'} else public_settings(item)) for key,item in value.items()}
    if isinstance(value,list):return [public_settings(item) for item in value]
    return value


def configuration_view(run, *, snapshot_id=None, test_id=None):
    run=Path(run).resolve();project=project_from_path(run)
    if project is None:raise ValueError('Project configuration is not available')
    store=HistoryStore(project);snapshot=store.snapshot(snapshot_id) if snapshot_id else None
    if snapshot_id and (not snapshot or snapshot['revision']!=run.name):raise ValueError('Unknown revision snapshot')
    sources=[];generation=None
    if snapshot:
        config=snapshot.get('config_path')
        if config:generation=read_json(config);sources.append(config)
    else:
        configs=[]
        for path in (project/'worker_turns').rglob('runtime_config.*.json'):
            value=read_json(path)
            if value.get('revision')==run.name:configs.append((value.get('captured_ms',0),path,value))
        if configs:
            _,path,generation=max(configs,key=lambda x:x[0]);sources.append(str(path))
    scene_path=(snapshot.get('files',{}).get('scene.json') if snapshot else str(run/'design/scene.json'))
    scene=read_json(scene_path) if scene_path else {}
    if scene_path and scene:sources.append(scene_path)
    keys=('parameters','units','gravity_m_s2','joints','tests','numerical_settings','equivalent_elasticity')
    mechanical={key:scene[key] for key in keys if key in scene}
    bodies=scene.get('bodies') if isinstance(scene.get('bodies'),list) else []
    mechanical['bodies']=[{key:body[key] for key in ('id','source','fixed','density_kg_m3','material') if key in body}
                          for body in bodies if isinstance(body,dict)]
    # These are saved geometry facts from the selected tool attempt, not fresh defaults.
    geometry_path=Path(snapshot['attempt'])/'geometry/geometry.json' if snapshot and snapshot.get('attempt') else None
    geometry=read_json(geometry_path) if geometry_path else {}
    if geometry:
        sources.append(str(geometry_path))
        mechanical['geometry_units']=geometry.get('units')
        mechanical['recorded_body_properties']=[{key:body[key] for key in ('id','mass_kg','volume_m3','density_kg_m3','position_m','rotation_wxyz','inertia_kg_m2','mass_properties_source') if key in body} for body in geometry.get('bodies',[])]
    if test_id and isinstance(mechanical.get('tests'),list):mechanical['tests']=[t for t in mechanical['tests'] if t.get('id')==test_id]
    simulations={}
    for path in (project/'trajectory/configurations').glob('*.json'):
        reference=read_json(path)
        if reference.get('revision')!=run.name or test_id and reference.get('test_id')!=test_id:continue
        reference_path=resolve_workspace_reference(project.parents[2],reference.get('file',''))
        value=read_json(reference_path)
        if snapshot:
            binding=value.get('history_binding') or {}
            saved_binding=snapshot.get('model_binding')
            if not saved_binding and snapshot.get('attempt'):
                from ..trajectory.configuration import model_binding
                saved_binding=model_binding(read_json(Path(snapshot['attempt'])/'geometry/geometry.json'))
            if value.get('snapshot_id')!=snapshot['id'] and not (binding.get('sources') and binding==saved_binding):continue
        if value:
            simulations[str(reference_path)]=value
            sources.append(str(reference_path))
    # Legacy requests are explicitly labelled requests, not inferred effective defaults.
    if not simulations and not snapshot:
        for path in run.glob('attempts/*/tests/*/request.json'):
            value=read_json(path);test=value.get('test') or {}
            if test_id and test.get('id')!=test_id:continue
            simulations[str(path)]={'kind':'request','test_id':test.get('id'),'limits':value.get('execution_limits'),
                                    'numerical':value.get('numerical_settings'),'source':str(path)}
    simulations=sorted(simulations.values(),key=lambda x:x.get('captured_ms',0),reverse=True)[:100]
    return public_settings({'revision':run.name,'snapshot':snapshot_id,'test_id':test_id,'generation':generation,
            'mechanical':mechanical if scene or geometry else None,'simulations':simulations,'sources':sources,
            'note':'未记录的字段不会以当前默认值补齐。仿真请求与实际应用值分别标注。','read_only':True})
