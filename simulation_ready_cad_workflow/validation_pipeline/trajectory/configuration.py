"""Immutable, credential-free execution settings for later inspection."""
import logging
import time
import uuid
from pathlib import Path

from .capture import read_json, write_json
from .store import project_from_path, revision_from_path

_BUDGET_KEYS=('hours','simulation_timeout_seconds','audit_timeout_seconds','max_tasks','max_tests',
              'max_geometry_units','max_geometry_units_per_body','summary_reserve_seconds')


def model_binding(geometry):
    """Reuse validator-owned source digests, including geometry changes with identical scene JSON."""
    parent=Path(str(geometry.get('scene_file') or '.').replace('\\','/')).parent
    sources=[]
    for source in geometry.get('source_bindings') or []:
        path=Path(str(source.get('source_file') or '').replace('\\','/'))
        try:name=path.relative_to(parent).as_posix()
        except ValueError:name=path.name
        if source.get('source_sha256'):sources.append([name,source['source_sha256']])
    return {'scene_sha256':geometry.get('scene_sha256'),'sources':sorted(sources)}


def worker_configuration(worker, project, workspace, readonly):
    providers={'CodexCliWorker':'codex','CodexAppWorker':'codex_app','CursorCliWorker':'cursor','KimiCliWorker':'kimi','ClaudeCodeWorker':'claude_code','AnthropicWorker':'anthropic'}
    data={'scope':'worker','revision':revision_from_path(workspace),'captured_ms':int(time.time()*1000),
          'worker':{'provider':providers.get(type(worker).__name__),'model':getattr(worker,'model',None),
                    'reasoning_effort':getattr(worker,'reasoning_effort',None),
                    'executable':getattr(worker,'executable',None),'readonly':bool(readonly)},'budget':{}}
    if project:
        value=read_json(Path(project)/'construction_budget.json').get('config') or {}
        data['budget']={k:value[k] for k in _BUDGET_KEYS if k in value}
        checkpoint=read_json(Path(project)/'budget_checkpoint.json')
        for key in ('tier','tier_config','budget_limit_seconds','base_budget_limit_seconds','granted_active_seconds','remaining_active_seconds'):
            if key in checkpoint:data['budget'][key]=checkpoint[key]
    return data


def save_simulation_configuration(folder, *, interpreter, test, limits=None, numerical=None, kind='request', history_binding=None):
    folder=Path(folder).resolve();project=project_from_path(folder)
    attempt=next((p for p in (folder,*folder.parents) if p.parent.name=='author_tools'),None)
    meta=read_json(attempt/'capture.json') if attempt else {}
    if project is None and meta.get('project'):project=Path(meta['project'])
    if project is None:return None
    values={'scope':'simulation','revision':revision_from_path(folder) or meta.get('revision'),'test_id':test.get('id'),
            'attempt':str(folder),'captured_ms':int(time.time()*1000),'kind':kind,
            'interpreter':str(interpreter),'limits':limits or {},'numerical':numerical or {},
            'duration_s':test.get('duration_s'),'test_step_s':test.get('step_s'),
            'history_binding':history_binding or {},'snapshot_id':attempt.name if attempt else None}
    try:
        target=folder/('runtime_config.json' if kind=='request' else 'effective_config.json')
        write_json(target,values)
        registry=project/'trajectory/configurations'/f"{uuid.uuid4().hex}.json"
        write_json(registry,{'file':str(target),'revision':values['revision'],'test_id':test.get('id'),'kind':kind})
        return str(target)
    except OSError as exc:
        logging.getLogger(__name__).warning('Could not save observed runtime configuration: %s',exc)
        return None
