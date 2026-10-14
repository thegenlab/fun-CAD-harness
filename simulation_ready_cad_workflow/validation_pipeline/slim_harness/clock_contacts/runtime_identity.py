"""Read identity of this loaded Chrono runtime only; never select another environment."""
from pathlib import Path
import argparse
import hashlib
import json
import sys


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def file_binding(path):
    path=Path(path).resolve()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha256_file(path)}


def runtime_identity(core):
    versions={}
    for name in ('CHRONO_VERSION','CHRONO_VERSION_MAJOR','CHRONO_VERSION_MINOR','CHRONO_VERSION_PATCH','CH_VERSION'):
        value=getattr(core,name,None)
        if isinstance(value,(str,int,float)):versions[name]=value
    getter=getattr(core,'GetChronoVersion',None)
    if callable(getter):versions['GetChronoVersion']=str(getter())
    package_records=[]
    # Metadata of the active interpreter prefix, not an environment search.
    metadata=Path(sys.prefix)/'conda-meta'
    if metadata.is_dir():
        for path in sorted(metadata.glob('*chrono*.json')):
            record=json.loads(path.read_text(encoding='utf-8'))
            package_records.append({key:record.get(key) for key in ('name','version','build','build_number')})
    loaded=set()
    maps=Path('/proc/self/maps')
    prefix=Path(sys.prefix).resolve()
    if maps.is_file():
        for line in maps.read_text().splitlines():
            pieces=line.split(maxsplit=5)
            if len(pieces)==6 and pieces[5].startswith('/'):
                path=Path(pieces[5])
                if path.is_file() and (path.resolve().is_relative_to(prefix) or 'chrono' in path.name.lower()):
                    loaded.add(path.resolve())
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith('pychrono') and path and Path(path).suffix in ('.so','.pyd','.dll'):
            loaded.add(Path(path).resolve())
    return {'schema_version':'chrono_loaded_runtime_identity_v1','python':sys.executable,
            'python_version':list(sys.version_info[:3]),'prefix':sys.prefix,
            'chrono_version_constants':versions,'active_chrono_package_records':package_records,
            'core_wrapper':file_binding(core.__file__),
            'loaded_runtime_libraries':[file_binding(path) for path in sorted(loaded)],
            'scope':'Currently loaded native libraries from the active prefix plus named Chrono components; this includes runtime dependencies, not only Chrono-owned code. FEA contact behavior is not certified by rigid reference runs.'}


def capture_runtime(core,output,sources):
    output=Path(output);snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    result=runtime_identity(core);result['executed_probe_sources']=[]
    for source in sources:
        source=Path(source).resolve();target=snapshot/source.name
        data=source.read_bytes()
        with target.open('xb') as stream:stream.write(data)
        result['executed_probe_sources'].append({'source_path':str(source),'snapshot_file':str(target.relative_to(output)),
                                               'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (output/'runtime_identity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.backend_root.resolve()))
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import load_pychrono_core
    result=runtime_identity(load_pychrono_core())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'identity_file':str(args.output),'versions':result['chrono_version_constants'],
                      'packages':result['active_chrono_package_records']},ensure_ascii=False))


if __name__=='__main__':main()
