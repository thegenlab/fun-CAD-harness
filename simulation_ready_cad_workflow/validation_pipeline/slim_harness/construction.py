"""Construction policy: adaptable proposed tests, immutable core task semantics.

Only projects with a construction budget opt in. A frozen experiment remains
an evaluation contract and cannot acquire these authoring permissions.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
import sys
import time
from pathlib import Path

from .construction_budget import ProjectBudget


def review_notes(project):
    """Read optional maintainer advice without changing any acceptance contract.

    Read at tool completion so an already-running author sees new advice on its
    next ordinary tool response. Missing/unreadable notes never block work.
    """
    if project is None:
        return None
    project = Path(project)
    if not (project / 'construction_budget.json').is_file():
        return None
    try:
        state = json.loads((project / 'workflow.json').read_text(encoding='utf-8'))
        if state.get('request', {}).get('frozen_experiment_sha256'):
            return None
        path = project / 'construction_review_notes.md'
        with path.open(encoding='utf-8', errors='replace') as stream:
            text = stream.read(4001)
    except (OSError, ValueError):
        return None
    if not text.strip():
        return None
    return {'source': str(path), 'text': text[:4000], 'truncated': len(text) > 4000,
            'advisory_only': True,
            'scope': 'Optional independent review advice, not a verdict, new acceptance condition or required response contract. Consider it alongside the current design and remaining budget.'}


def budget_for(project):
    if project is None:return None
    path=Path(project)
    # The ledger owns this marker, not a scene authored by a model.
    if not (path/'construction_budget.json').is_file():return None
    from .frozen_experiment import load_experiment
    if load_experiment(path) is not None:
        raise ValueError('Construction permissions cannot modify a frozen experiment')
    return ProjectBudget(path)


def construction_requested(project):
    """Supervised construction, not a LIGHT closeout ledger alone."""
    if project is None:
        return False
    try:
        state=json.loads(Path(project).joinpath('workflow.json').read_text(encoding='utf-8'))
    except (OSError,ValueError,TypeError):
        return False
    return bool(isinstance(state,dict) and state.get('request',{}).get('construction'))


def policy_prompt(project):
    budget=budget_for(project)
    if budget is None:return ''
    state=budget.snapshot();config=state['config']
    from .construction_limits import simulation_time_steps
    steps=simulation_time_steps(config["simulation_timeout_seconds"])
    from .project_isolation import CODE_ROOT
    root=Path(project).parents[2]
    pythonpath=str(CODE_ROOT)
    if (root/'.runtime_runs/python_deps').is_dir():pythonpath+=os.pathsep+str(root/'.runtime_runs/python_deps')
    prompt = (
        '\nBENCHMARK CONSTRUCTION MODE (not frozen benchmark evaluation).\n'
        f'Exact CLI prefix from any working directory: PYTHONPATH={pythonpath} {sys.executable} -m simulation_ready_cad_workflow.validation_pipeline.slim_harness --workspace {root} . '
        f'Budget query: append budget status --project {Path(project).name}. '
        f'Effective project budget remaining: {state["remaining_seconds"]:.1f}s. Strategy: {state["strategy_tier"]}. '
        f'Maximum {config["max_tasks"]} mechanical task groups and {config["max_tests"]} scene tests; '
        +(f'each simulation has a HARD {steps[0]:g}s wall timeout. ' if len(steps)==1 else
          f'a simulation still running at {steps[0]:g}s is extended to '+' and then '.join(f'{step:g}s' for step in steps[1:])+', its HARD wall timeout. ')+
        'A simulation command you wait on is not charged to this budget; work you do while a background simulation runs is charged, as is all other author, reviewer, tool and Harness work; use budget status before costly work. '
        'Build a recognizable complete minimum asset early, then test all core functions. '
        'In construction_plan.json identify mandatory core functions/numbers, permitted test adjustments and supported physical abstractions. '
        'Keep source CAD detail, but group components with no relative motion into rigid bodies and retain only function-relevant contacts. '
        'Before an expensive new simulation, use probe --test to measure a short prefix, inspect its cost estimate and decide whether to change the design. '
        'A cost probe is diagnostic only: it never covers a declared test or supplies acceptance evidence. '
        'Do not repeat an unchanged timed-out workload; inspect cost and failure evidence and record a reasoned change. '
        'After failure choose among (1) repair mechanical design, (2) simplify simulation using flexible/finite '
        'or abstract joints plus original static geometry checks, (3) relax a proposed test within unchanged core semantics. '
        'You may edit scene test duration, sampling, secondary loads/tolerances and modeling fidelity if the task permits it; '
        'record the original and revised plan, reason and unsupported claims in design/construction_plan.json. '
        'Do not silently remove core functions, replace transmission with directly driving its output, invent success, '
        'or change explicit mandatory task numbers. No controller trajectories alone prove passive mechanics. '
        'Explain and use the installed equivalent_elasticity / supported joints / spur_gears interfaces when suitable. '
        'If a required abstraction is not implemented, record the capability gap and provide static evidence as partial, '
        'rather than claiming a physical test passed. Source visual detail need not enter microscopic contact simulation. '
        'Task-change review supplies advice; program checks remain binding and evidence must state the actual scope. '
        'As time becomes tight, stop adding tests and cosmetic refinements; prioritize core coverage and evidence. '
        'During finalizing, deliver the latest build and record partial work/reasons/recommendations immediately. '
        'Enhance strictness or add optional tests only after the minimum core tests pass and sufficient time remains. '
        'Use CLI budget pause/resume for environment_setup or human_intervention; never pause to hide ordinary design/physics time. '
        'Write no Harness PASS or completion receipt yourself. The Harness always emits completion_summary.json/md.\n'
    )
    notes = review_notes(project)
    from .modes import settings as mode_settings
    if mode_settings(project):
        prompt += ('\nIn template modes, any scene.tests are construction diagnostics only. '
                   'The Harness reviewer fills the formal task-template binding. Request formal difficulty '
                   'changes through template_adjustment.json; do not fill your own binding or modify public templates.\n')
    if notes:
        prompt += '\nOPTIONAL CONSTRUCTION REVIEW ADVICE:\n' + notes['scope'] + '\n' + notes['text'] + '\n'
    return prompt


def revision_choices(project,result):
    if budget_for(project) is None:return result
    return {**result,'failure_cause':'not_uniquely_determined',
        'available_repair_strategies':['modify_mechanics','simplify_simulation','relax_noncore_test_strictness'],
        'construction_scope':'Core criteria remain unchanged. Simplification/test changes must be declared and reviewed.',
        'budget':budget_for(project).snapshot()}


def worker_for(project,worker):
    """Attach shared accounting without changing frozen/legacy worker callers."""
    if budget_for(project) is not None and construction_requested(project):
        worker.construction_project=Path(project)
    return worker


def author_turn_timeout(project,default_seconds=1800):
    """One construction clock, not a second arbitrary author restart limit.

    Review/tool limits remain separate. The external supervisor and the worker
    still enforce the project's effective-work deadline, including native calls.
    A LIGHT closeout ledger does not replace the ordinary author turn limit.
    """
    budget=budget_for(project)
    if budget is None or not construction_requested(project):
        return default_seconds
    from .test_coverage import unvisited,planned_tests,FirstPassReserveReached
    snap=budget.check_available()
    tests=planned_tests(project)
    pending=len(unvisited(project,tests)) if tests is not None else snap['config']['max_tests']
    available=snap['work_remaining_seconds']-pending*snap['config']['simulation_timeout_seconds']
    if available<=0:raise FirstPassReserveReached("Construction author time is reserved for untested tasks")
    return available


def session_id_from_event(event):
    if not isinstance(event, dict):
        return None
    if event.get('type')=='thread.started' and isinstance(event.get('thread_id'), str) and event['thread_id']:
        return event['thread_id']
    session=event.get('session_id')
    if isinstance(session, str) and session:
        return session
    return None


def _saved_formal_session(project,client,selected=None):
    from ..benchmark_sandbox.formal_state import load_seed,session_id
    records=project/'.benchmark-sandbox/sessions'/client/'author-persistent'
    path=records/((session_id(selected)+'.json') if selected else 'current.json')
    if not path.is_file():return None
    if path.is_symlink():raise ValueError('Formal session record must be an ordinary private file')
    record=json.loads(path.read_text(encoding='utf-8'));sid=session_id(record['session_id'])
    if selected and sid!=selected:raise ValueError('Saved formal author session identity changed')
    payload=Path(record['session_state_path']).resolve(strict=True)
    if (payload.name!='session-state' or payload.parent.name!='sandbox-run'
            or payload.parent.parent.name!='.benchmark-sandbox'
            or not payload.parent.parent.parent.name.endswith('_author')
            or payload.parent.parent.parent.parent!=project/'worker_turns'):
        raise ValueError('Formal author state must belong to its own private author turn')
    load_seed(payload,client,sid)
    if client=='cursor':
        if not (payload/'cursor/store.db').is_file():raise ValueError('Saved formal Cursor database is absent')
    else:
        rollouts=list((payload/'codex/sessions').rglob('*.jsonl'))
        if len(rollouts)!=1:raise ValueError('Saved formal Codex rollout is not unique')
        first=json.loads(rollouts[0].read_bytes().splitlines()[0])
        if first.get('type')!='session_meta' or first.get('payload',{}).get('id')!=sid:
            raise ValueError('Saved formal Codex rollout identity changed')
    return sid,path


def _only_formal_init(project,source,sid,client):
    """Recognize only the old auto-inference bug, never discard completed state."""
    path=Path(source).resolve()
    if (client!='cursor' or path.name!='events.jsonl' or path.parent.parent!=project/'worker_turns'
            or not path.parent.name.endswith('_author') or not (path.parent/'.benchmark-sandbox').is_dir()):
        return False
    try:events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError,ValueError):return False
    if not events or not any(session_id_from_event(row)==sid for row in events):return False
    if any(row.get('type') not in {'system','user'} or
           row.get('type')=='system' and row.get('subtype')!='init' for row in events):return False
    for turn in (project/'worker_turns').glob('t*_author'):
        receipt=turn/'sandbox-receipt.json'
        if receipt.is_file():
            value=json.loads(receipt.read_text(encoding='utf-8'))
            if value.get('session_id')==sid and (value.get('client_completed') or value.get('session_state_preserved')):
                return False
        log=turn/'events.jsonl'
        if not log.is_file():continue
        rows=[]
        for line in log.read_text(encoding='utf-8',errors='replace').splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            rows.append(event)
        if any(session_id_from_event(row)==sid for row in rows) and any(
                row.get('type')=='turn.completed' or row.get('type')=='result'
                and not row.get('is_error') and row.get('subtype') not in {'error','failed'} for row in rows):
            return False
    return True


def recover_author_session(project,state):
    """CLI emits its session ID before a turn finishes; survive host interruption."""
    if budget_for(project) is None:return state
    project=Path(project).resolve()
    formal=any((turn/'.benchmark-sandbox').is_dir() or (turn/'sandbox-receipt.json').is_file()
               for turn in (project/'worker_turns').glob('t*_author'))
    if formal:
        client=state.get('request',{}).get('worker_provider','codex')
        client={'cursor_cli':'cursor','cursor_agent':'cursor'}.get(client,client)
        if client not in {'codex','cursor'}:return state
        selected=state.get('author_session_id')
        if selected:
            saved=_saved_formal_session(project,client,selected)
            if saved:return state
            source=state.get('author_session_recovered_from')
            if not source or not _only_formal_init(project,source,selected,client):return state
            private=project/'.benchmark-sandbox/session-recovery';private.mkdir(mode=0o700,parents=True,exist_ok=True)
            backup=private/('workflow-before-'+str(time.time_ns())+'.json')
            with backup.open('xb') as stream:stream.write((project/'workflow.json').read_bytes())
            backup.chmod(0o600)
            state['author_session_recovery_rejected']={'session_id':selected,'source':source,
                'reason':'auto_inferred_init_only_without_native_state','preserved_workflow':str(backup)}
            state['author_session_id']=None
            state.pop('author_session_recovered_from',None)
        # A completed original turn cannot be replaced by a fresh or older session
        # merely because its state export/index is missing.
        for turn in (project/'worker_turns').glob('t*_author'):
            if not (turn/'.benchmark-sandbox').is_dir():continue
            from ..benchmark_sandbox.formal_state import original_events,observed_session
            rows=original_events(turn/'events.jsonl',allow_incomplete=True)
            complete=any(row.get('type')=='turn.completed' or row.get('type')=='result'
                         and not row.get('is_error') and row.get('subtype') not in {'error','failed'} for row in rows)
            if complete and _saved_formal_session(project,client,observed_session(client,rows)) is None:
                raise ValueError('Completed formal author session has no verified saved state; preserve it for recovery')
        saved=_saved_formal_session(project,client)
        if saved:
            state['author_session_id']=saved[0]
            state['author_session_recovered_from']=str(saved[1])
        return state
    for turn in sorted((Path(project)/'worker_turns').glob('t*_author'),reverse=True):
        events=turn/'events.jsonl'
        if not events.exists():continue
        for line in events.read_text(encoding='utf-8',errors='replace').splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            session=session_id_from_event(event)
            if session:
                state['author_session_id']=session
                state['author_session_recovered_from']=str(events)
                return state
    return state


def activity(project,kind):
    budget=budget_for(project)
    return budget.activity(kind) if budget else nullcontext()


def check(project):
    budget=budget_for(project)
    if budget:budget.check_available()


def summary(project,*,reason='',result=None):
    from .construction_completion import write_completion_summary
    return write_completion_summary(project,reason=reason,result=result)
