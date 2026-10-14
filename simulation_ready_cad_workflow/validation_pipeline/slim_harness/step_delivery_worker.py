"""Bounded fresh Asset rebuild and STEP inspection; no author PASS is consumed."""
import json
from pathlib import Path
import runpy
import shutil
import sys

from .asset import build_file
from .static_delivery import binding,read,write
from .static_delivery_worker import facts,consume_matches


def inspect(payload):
    import build123d as b
    import OCP
    design,rebuilt=Path(payload['design']),Path(payload['rebuilt_design'])
    request=payload['request'];scene=read(design/'scene.json')
    rebuilt.mkdir(exist_ok=False)
    for name in request['source_files']:
        if name in ('scene.json','parts.json') or Path(name).suffix.lower() in ('.step','.stp'):continue
        target=rebuilt/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(design/name,target)
    source=rebuilt/request['model_source']
    sys.path.insert(0,str(source.parent))
    try:
        namespace=runpy.run_path(str(source))
        if not callable(namespace.get('build')) or not callable(namespace.get('gen_step')):
            raise ValueError('The declared source must provide build() and gen_step() for the delivered Asset and complete assembly')
        shape=namespace['gen_step']()
        assembly=rebuilt/request['assembly_step'];assembly.parent.mkdir(parents=True,exist_ok=True)
        if not b.export_step(shape,assembly):raise ValueError('No fresh assembly STEP generated')
        build_file(source,rebuilt)
    finally:sys.path.pop(0)
    fresh=read(rebuilt/'scene.json')
    for field in ('schema_version','units','bodies','joints','tests','gravity_m_s2','equivalent_elasticity'):
        if scene.get(field)!=fresh.get(field):raise ValueError('Rebuilt scene differs: '+field)
    if read(design/'parts.json')!=read(rebuilt/'parts.json'):raise ValueError('Rebuilt physical-parts inventory differs')
    measured=[]
    for name in dict.fromkeys([request['assembly_step'],*(body['source'] for body in scene['bodies'])]):
        before=list(b.import_step(design/name).solids());after=list(b.import_step(rebuilt/name).solids())
        if not before or not after:raise ValueError('STEP contains no solids: '+name)
        pairs=consume_matches([facts(s) for s in before],[facts(s) for s in after])
        measured.append({'source':name,'solid_count':len(before),'correspondence':pairs})
    for name in request['documents']:
        if not (design/name).read_text(encoding='utf-8-sig').strip():raise ValueError('Empty delivery document: '+name)
    return {'completed':True,'scope':'Fresh source execution; STEP leaf inventory correspondence within the existing static verifier tolerances, not a Boolean equivalence proof or a mechanical verdict.',
            'step_files':measured,'physical_part_inventory_reproduced':True,'physical_scene_reproduced':True,
            'documents':request['documents'],'documentation_semantics_verified':False,'print_assets_required':False,
            'runtime_bindings':binding([sys.executable,b.__file__,OCP.__file__]),
            'rebuilt_files':[str(p) for p in rebuilt.rglob('*') if p.is_file()]}


if __name__=='__main__':
    payload=read(sys.argv[1]);write(payload['output'],inspect(payload))
