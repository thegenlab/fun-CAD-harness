"""Small durable stage markers; never a physical result or a certificate."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


class RuntimeProgress:
    def __init__(self, output):
        self.path=Path(output).with_suffix('.progress.json')
        self.log_path=Path(output).with_suffix('.progress.jsonl')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic()
        self.last_write=-float('inf')
        self.state={'schema_version':'slim_runtime_progress_v1','pid':os.getpid(),
            'trace_file':str(output),'certified':False,'recorded_samples':0,
            'last_recorded_time_s':None}

    def mark(self, stage, *, force=False, **details):
        changed=stage!=self.state.get('stage')
        self.state.update(stage=stage,**details)
        now=time.monotonic()
        if not (force or changed or now-self.last_write>=1.):return
        self.state.update(updated_at=datetime.now(timezone.utc).isoformat(),wall_seconds=now-self.started)
        encoded=json.dumps(self.state,ensure_ascii=False,allow_nan=False)
        temporary=self.path.with_name(self.path.name+f'.{os.getpid()}.tmp')
        temporary.write_text(encoded+'\n',encoding='utf-8')
        temporary.replace(self.path)
        with self.log_path.open('a',encoding='utf-8') as log:
            log.write(encoded+'\n');log.flush()
        self.last_write=now


def worker_start(request_path):
    """Called before native/runtime imports, so a startup stall has a marker."""
    value=json.loads(Path(request_path).read_text(encoding='utf-8'))
    progress=RuntimeProgress(value['output'])
    progress.mark('importing_runtime',request_file=str(request_path),
                  requested_duration_s=value['test']['duration_s'])
    return progress
