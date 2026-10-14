"""Work that follows the outer public run call after it has released its workers."""
from contextvars import ContextVar
from functools import wraps
from importlib.util import find_spec
import os
from pathlib import Path

from ..snapshot_paths import imported

_ARCHIVE = __package__.rsplit('.', 1)[0] + '.hf_sync'
_depth = ContextVar('harness_public_run_depth', default=0)


def after_run(function):
    @wraps(function)
    def wrapped(workspace, *args, **kwargs):
        project = args[0] if args else kwargs.get('project_id', kwargs.get('project'))
        root = Path(workspace).resolve()
        if imported(root):
            raise ValueError('Imported snapshots are historical; create a new local project to continue work')
        outer = _depth.get() == 0 and not kwargs.get('in_process') and not os.environ.get('SIMREADY_CONSTRUCTION_PROJECT')
        token = _depth.set(_depth.get() + 1)
        try:
            return function(workspace, *args, **kwargs)
        finally:
            _depth.reset(token)
            # Team archiving is an optional private package.
            if outer and find_spec(_ARCHIVE) is not None:
                from ..hf_sync.hooks import publish_if_configured
                publish_if_configured(root, project)
    return wrapped
