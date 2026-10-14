"""Independent publisher of persisted public logs and retained draft geometry."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

from .capture import FileObserver, TurnCapture, now_ms, read_json, snapshot_attempt, write_json
from .store import revision_from_path


def owner_running(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        try:
            code=wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except OSError:return False


def publish_pending(project, seen):
    """Read only attempts explicitly queued for this project; never infer a later revision."""
    for request in sorted((Path(project)/'trajectory/pending').glob('*.json')):
        job=read_json(request)
        if not job.get('attempt'):continue
        attempt=Path(job['attempt']);meta=read_json(attempt/'capture.json')
        if meta.get('project')!=str(project):continue
        # Producer metadata is atomic; the stage/result signature changes even on exFAT.
        signature=(meta.get('publication_stage'),meta.get('geometry_ready_ms'),meta.get('completed_ms'))
        if seen.get(str(attempt))!=signature:
            snapshot_attempt(job['workspace'],attempt,meta.get('publish_result',{}))
            seen[str(attempt)]=signature
        if meta.get('completed_ms'):
            request.unlink(missing_ok=True)


def run_attempt(path):
    seen={}
    while True:
        meta=read_json(path)
        if not meta.get('project'):return
        publish_pending(Path(meta['project']),seen)
        if meta.get('completed_ms') or not owner_running(meta.get('owner_pid')):return
        time.sleep(.5)


def run_turn(path):
    control=read_json(path)
    capture=TurnCapture(control['design'],control['output'],readonly=control.get('readonly',False))
    if not capture.store:return
    capture.revision=control['revision']
    if not capture.readonly and revision_from_path(capture.design):
        capture.observer=FileObserver(capture.store,capture.design,capture.turn)
        capture.observer.revision=capture.revision
        capture.observer.config_path=control.get('config_path')
    seen={}
    try:
        while True:
            control=read_json(path)
            final=control.get('status')!='recording' or not owner_running(control.get('owner_pid'))
            publish_pending(capture.project,seen)
            capture._tick(final=final)
            if final:break
            time.sleep(.5)
        capture.store.set_state('capture',{'status':'interrupted' if control.get('status')!='finished' else 'idle',
            'turn':capture.turn,'updated_ms':now_ms(),'error':None,'cpu_seconds':capture.cpu_seconds,
            'publisher_pid':os.getpid()})
    except Exception as exc:
        write_json(capture.output/'capture-error.json',{'error':str(exc),'at_ms':now_ms()})
        raise


def main():
    mode,path=sys.argv[1:]
    if mode=='turn':run_turn(Path(path))
    elif mode=='attempt':run_attempt(Path(path))
    else:raise ValueError('Unknown publication request')


if __name__=='__main__':main()
