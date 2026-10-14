"""Small SQLite index over immutable snapshots and append-only public logs."""
from __future__ import annotations

from contextlib import contextmanager
import difflib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

from .events import event_rows, event_ms, merge_rows, display_text, completed_tool_output

PAGE_BYTES = 512 * 1024
PAGE_ITEMS = 100
_ATTEMPT = re.compile(r"author_tools[/\\]+([a-z]+-[\w-]+)")
_ATTEMPT_FIELD = re.compile(r'["\\]+attempt["\\]+\s*:\s*["\\]+[^"\n]*?author_tools[/\\]+([a-z]+-[\w-]+)')


def project_from_path(path):
    path = Path(path).resolve()
    return next((p for p in (path, *path.parents)
                 if p.parent.name == 'projects' and p.parent.parent.name == '.runtime_runs'), None)


def revision_from_path(path):
    parts = Path(path).parts
    for i, part in enumerate(parts[:-1]):
        if part == 'runs' and re.fullmatch(r'r\d+', parts[i + 1]):
            return parts[i + 1]
    return ''


class HistoryStore:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.root = self.project / 'trajectory'
        self.path = self.root / 'index.sqlite3'
        from ..snapshot_paths import imported
        if imported(self.project.parents[2]):
            original = self.path
            self.path = self.project.parents[2] / '.runtime_runs/hf_view' / self.project.name / 'index.sqlite3'
            if original.is_file() and not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(original.as_uri() + '?mode=ro', uri=True) as src, sqlite3.connect(self.path) as dst:
                    src.backup(dst)

    @contextmanager
    def connect(self, write=False):
        if write:
            self.root.mkdir(parents=True, exist_ok=True)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, timeout=10)
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, raw_path TEXT NOT NULL, offset INTEGER NOT NULL,
                    length INTEGER NOT NULL, turn TEXT NOT NULL, at_ms INTEGER,
                    observed_ms INTEGER, payload TEXT NOT NULL, UNIQUE(raw_path,offset));
                CREATE INDEX IF NOT EXISTS events_turn_time ON events(turn,at_ms,seq);
                CREATE TABLE IF NOT EXISTS cursors (path TEXT PRIMARY KEY, offset INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS segments (path TEXT PRIMARY KEY, id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (
                    seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, revision TEXT NOT NULL,
                    turn TEXT, event_seq INTEGER, at_ms INTEGER, capture TEXT NOT NULL,
                    status TEXT NOT NULL, model TEXT, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS snapshots_revision ON snapshots(revision,seq);
                CREATE TABLE IF NOT EXISTS attempt_events (
                    id TEXT PRIMARY KEY, event_seq INTEGER, turn TEXT, at_ms INTEGER);
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS assets (path TEXT PRIMARY KEY,bytes INTEGER NOT NULL);
                PRAGMA user_version=1;
                COMMIT;
            ''')
        else:
            if not self.path.is_file():
                yield None
                return
            db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=10)
            if db.execute('PRAGMA user_version').fetchone()[0] == 0:
                db.close()
                yield None
                return
        db.row_factory = sqlite3.Row
        try:
            yield db
            if write:
                db.commit()
        finally:
            db.close()

    def set_state(self, key, value):
        with self.connect(True) as db:
            db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', (key, json.dumps(value)))

    def state(self, key, default=None):
        with self.connect() as db:
            row = db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone() if db else None
        return json.loads(row[0]) if row else default

    def archive_log(self, source, target):
        with self.connect(True) as db:
            db.execute('UPDATE events SET raw_path=? WHERE raw_path=?', (str(target), str(source)))
            db.execute('UPDATE cursors SET path=? WHERE path=?', (str(target), str(source)))
            db.execute('UPDATE segments SET path=? WHERE path=?', (str(target), str(source)))

    def index_log(self, path, turn, *, observed=False, drain=False):
        """Parse new records once, outside HTTP requests. Keep unfinished tail for retry."""
        self.tool_completed=False
        path = Path(path).resolve()
        if not path.is_file():
            return 0
        total = 0
        while True:
            with self.connect(True) as db:
                row = db.execute('SELECT offset FROM cursors WHERE path=?', (str(path),)).fetchone()
                if row is None:
                    from ..dashboard.path_resolution import resolve_workspace_reference
                    for saved in db.execute('SELECT path,offset FROM cursors').fetchall():
                        try:relocated=resolve_workspace_reference(self.project.parents[2],saved['path'],require_within_workspace=True)
                        except ValueError:continue
                        if relocated!=path:continue
                        # The copied log keeps its byte cursor and event identities.
                        db.execute('UPDATE events SET raw_path=? WHERE raw_path=?',(str(path),saved['path']))
                        db.execute('UPDATE cursors SET path=? WHERE path=?',(str(path),saved['path']))
                        db.execute('UPDATE segments SET path=? WHERE path=?',(str(path),saved['path']))
                        row=saved;break
                offset = int(row['offset']) if row else 0
                db.execute('INSERT OR IGNORE INTO segments VALUES (?,?)',(str(path),uuid.uuid4().hex))
                segment=db.execute('SELECT id FROM segments WHERE path=?',(str(path),)).fetchone()[0]
                if path.stat().st_size < offset:
                    return total  # A truncated legacy log is not a new recording segment.
                consumed = 0
                with path.open('rb') as stream:
                    stream.seek(offset)
                    while consumed < PAGE_BYTES:
                        line = stream.readline()
                        if not line or not line.endswith(b'\n'):
                            break
                        position = offset + consumed
                        consumed += len(line)
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeError):
                            continue
                        rows = event_rows(event, position)
                        stamp = event_ms(event)
                        arrival = int(time.time() * 1000) if observed else None
                        if rows:
                            if (event.get('type') in {'item.completed','user'} or event.get('method')=='item/completed' or event.get('subtype')=='completed') and any(r.get('kind')=='tool' for r in rows):
                                self.tool_completed=True
                            # Public cards are bounded; the original full result stays in the raw log.
                            for item in rows:
                                if item.get('key'):
                                    item['key'] = f'{turn}:{segment}:{item["key"]}'
                                    item['event_key']=item['key']
                                    if 'title' not in item:
                                        # A Claude tool_result names only its call. Pages and live
                                        # polls replace rows by key, so carry the call's title and input.
                                        found = db.execute('SELECT payload FROM events WHERE raw_path=? AND instr(payload,?) ORDER BY offset DESC LIMIT 1',
                                                           (str(path), '"key": '+json.dumps(item['key']))).fetchone()
                                        call = next((r for r in json.loads(found[0]) if r.get('key')==item['key'] and 'title' in r), {}) if found else {}
                                        item.update({name:call[name] for name in ('title','text') if name in call})
                                if not item.get('at') and (stamp is not None or arrival is not None):
                                    item['at_ms'] = stamp if stamp is not None else arrival
                            effective = stamp if stamp is not None else arrival
                            db.execute('INSERT OR IGNORE INTO events(raw_path,offset,length,turn,at_ms,observed_ms,payload) VALUES (?,?,?,?,?,?,?)',
                                (str(path), position, len(line), turn, effective, arrival, json.dumps(rows)))
                            seq = db.execute('SELECT seq FROM events WHERE raw_path=? AND offset=?', (str(path), position)).fetchone()[0]
                            # Tool results contain the exact attempt path, including legacy JSON-in-text results.
                            output = completed_tool_output(event)
                            for name in set(_ATTEMPT_FIELD.findall(output)):
                                db.execute('INSERT OR IGNORE INTO attempt_events VALUES (?,?,?,?)', (name, seq, turn, effective))
                                db.execute('UPDATE snapshots SET event_seq=COALESCE(event_seq,?),turn=COALESCE(NULLIF(turn,\'\'),?),at_ms=COALESCE(at_ms,?) WHERE id=?',
                                           (seq, turn, effective, name))
                            total += 1
                db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?)', (str(path), offset + consumed))
            if not drain or consumed < PAGE_BYTES:
                return total

    def put_snapshot(self, snapshot):
        snapshot = dict(snapshot)
        with self.connect(True) as db:
            from ..dashboard.path_resolution import resolve_workspace_reference
            previous = db.execute('SELECT * FROM snapshots WHERE id=?', (snapshot['id'],)).fetchone()
            references=set(snapshot.get('files',{}).values())
            if snapshot.get('model'):references.add(snapshot['model'])
            old=json.loads(previous['data']) if previous else {}
            prior_refs=set(old.get('files',{}).values())
            if old.get('model'):prior_refs.add(old['model'])
            for reference in references | prior_refs:
                path=resolve_workspace_reference(self.project.parents[2],reference)
                if path.is_file():
                    db.execute('INSERT OR IGNORE INTO assets VALUES (?,?)',(str(path),path.stat().st_size))
                    if str(path)!=reference:db.execute('DELETE FROM assets WHERE path=?',(reference,))
            link = db.execute('SELECT * FROM attempt_events WHERE id=?', (snapshot['id'],)).fetchone()
            if link:
                snapshot['event_seq'] = link['event_seq']
                snapshot['turn'] = snapshot.get('turn') or link['turn']
                snapshot['at_ms'] = snapshot.get('at_ms') or link['at_ms']
            if previous:
                snapshot['event_seq'] = snapshot.get('event_seq') or previous['event_seq']
                snapshot['turn'] = snapshot.get('turn') or previous['turn']
                snapshot['at_ms'] = snapshot.get('at_ms') or previous['at_ms']
            data = json.dumps(snapshot, ensure_ascii=False)
            values = (snapshot['revision'], snapshot.get('turn'), snapshot.get('event_seq'), snapshot.get('at_ms'),
                      snapshot['capture'], snapshot['status'], snapshot.get('model'), data)
            if previous:
                db.execute('UPDATE snapshots SET revision=?,turn=?,event_seq=?,at_ms=?,capture=?,status=?,model=?,data=? WHERE id=?', (*values,snapshot['id']))
            else:
                db.execute('INSERT INTO snapshots(id,revision,turn,event_seq,at_ms,capture,status,model,data) VALUES (?,?,?,?,?,?,?,?,?)', (snapshot['id'],*values))

    def snapshot_row(self, row):
        if not row:
            return None
        result = json.loads(row['data'])
        result.update(seq=row['seq'], turn=row['turn'], event_seq=row['event_seq'], at_ms=row['at_ms'])
        # Resolve copied references for readers; the recorded SQLite payload
        # and its source/evidence bindings remain unchanged.
        from ..dashboard.path_resolution import resolve_workspace_reference
        workspace = self.project.parents[2]
        for key in ('model', 'source', 'attempt', 'config_path'):
            if result.get(key):
                result[key] = str(resolve_workspace_reference(workspace, result[key]))
        if result.get('files'):
            result['files'] = {name: str(resolve_workspace_reference(workspace, path))
                               for name, path in result['files'].items()}
        return result

    def snapshot(self, identity):
        with self.connect() as db:
            row = db.execute('SELECT * FROM snapshots WHERE id=?', (identity,)).fetchone() if db else None
        return self.snapshot_row(row)

    def recent_builds(self, revision):
        with self.connect() as db:
            rows=db.execute("SELECT * FROM snapshots WHERE revision=? AND capture='build' ORDER BY seq DESC LIMIT 3",(revision,)).fetchall() if db else []
        return [self.snapshot_row(row) for row in rows]

    def preceding_snapshots(self, snapshot, limit=100, *, across_revisions=False):
        with self.connect() as db:
            if db is None:return []
            if snapshot.get('at_ms') is not None:
                rows=db.execute("SELECT * FROM snapshots WHERE revision=? AND (at_ms<? OR (at_ms=? AND seq<?)) AND status!='incomplete' AND (model IS NOT NULL OR json_extract(data,'$.source') IS NOT NULL) ORDER BY at_ms DESC,seq DESC LIMIT ?",
                    (snapshot['revision'],snapshot['at_ms'],snapshot['at_ms'],snapshot['seq'],limit)).fetchall()
            elif snapshot.get('event_seq') is not None:
                rows=db.execute("SELECT * FROM snapshots WHERE revision=? AND event_seq<? AND status!='incomplete' AND (model IS NOT NULL OR json_extract(data,'$.source') IS NOT NULL) ORDER BY event_seq DESC LIMIT ?",
                    (snapshot['revision'],snapshot['event_seq'],limit)).fetchall()
            else:
                rows=[]  # Unordered records do not establish a previous model within the revision.
            if across_revisions and len(rows)<limit:
                rows.extend(db.execute("SELECT * FROM snapshots WHERE CAST(SUBSTR(revision,2) AS INTEGER)<? AND status!='incomplete' AND (model IS NOT NULL OR json_extract(data,'$.source') IS NOT NULL) ORDER BY CAST(SUBSTR(revision,2) AS INTEGER) DESC,revision DESC,at_ms IS NOT NULL DESC,at_ms DESC,COALESCE(event_seq,seq) DESC,seq DESC LIMIT ?",(int(snapshot['revision'][1:]),limit-len(rows))).fetchall())
        return [self.snapshot_row(row) for row in rows]

    def history(self, revision, *, around=None, limit=100, before=None, around_id=None):
        limit = min(200, max(1,int(limit)))
        with self.connect() as db:
            if db is None:
                return {'items':[], 'total':0, 'offset':0, 'latest_seq':0, 'indexing':True}
            total = db.execute('SELECT COUNT(*) FROM snapshots WHERE revision=?',(revision,)).fetchone()[0]
            selected_ordinal=None
            if around_id:
                found=db.execute('SELECT ordinal FROM (SELECT id,ROW_NUMBER() OVER (ORDER BY at_ms IS NOT NULL,at_ms,COALESCE(event_seq,seq),seq)-1 AS ordinal FROM snapshots WHERE revision=?) WHERE id=?',(revision,around_id)).fetchone()
                selected_ordinal=found[0] if found else None
                if around is None:around=selected_ordinal
            offset = max(0, total-limit) if around is None else max(0, min(total-limit,int(around)-limit//2))
            if before is not None:
                offset = max(0,int(before)-limit)
            rows = db.execute('SELECT * FROM snapshots WHERE revision=? ORDER BY at_ms IS NOT NULL,at_ms,COALESCE(event_seq,seq),seq LIMIT ? OFFSET ?', (revision,limit,offset)).fetchall()
            latest = db.execute('SELECT COALESCE(MAX(seq),0) FROM events').fetchone()[0]
        return {'items':[self.snapshot_row(r) for r in rows], 'total':total, 'offset':offset, 'latest_seq':latest, 'indexing':False,'selected_ordinal':selected_ordinal}

    def storage(self):
        with self.connect() as db:
            if db is None:return {'original_asset_bytes':0,'indexed_log_bytes':0}
            assets=db.execute('SELECT COALESCE(SUM(bytes),0) FROM assets').fetchone()[0]
            logs=db.execute('SELECT COALESCE(SUM(offset),0) FROM cursors').fetchone()[0]
        return {'original_asset_bytes':assets,'indexed_log_bytes':logs}

    def event_page(self, *, turn=None, before=None, after=None, from_ms=None, to_ms=None, snapshot=None):
        selected = self.snapshot(snapshot) if snapshot else None
        if snapshot and selected is None:
            raise ValueError('Unknown history snapshot')
        if selected:
            turn = selected.get('turn')
            if not turn:
                return {'items':[], 'turn':'', 'next_before':None, 'events_recorded':False,
                        'association':'unknown', 'message':'此快照没有可确认的作者轮次；可清除快照选择查看项目日志。'}
        clauses, params = [], []
        for condition, value in [('turn=?',turn), ('seq<?',before), ('seq>?',after), ('at_ms>=?',from_ms), ('at_ms<?',to_ms)]:
            if value is not None and value != '':
                clauses.append(condition); params.append(value)
        if selected and selected.get('event_seq') and before is None and after is None:
            clauses.append('seq<=?'); params.append(selected['event_seq'] + 15)
        with self.connect() as db:
            if db is None:
                return {'items':[], 'turn':turn or '', 'next_before':None, 'events_recorded':False}
            if not turn and not selected:
                last = db.execute('SELECT turn FROM events ORDER BY seq DESC LIMIT 1').fetchone()
                if last:
                    turn = last[0]; clauses.append('turn=?');params.append(turn)
            where = ' WHERE '+' AND '.join(clauses) if clauses else ''
            rows = db.execute('SELECT * FROM events'+where+' ORDER BY seq '+('ASC' if after is not None else 'DESC')+' LIMIT ?', (*params,PAGE_ITEMS)).fetchall()
            rows = sorted(rows,key=lambda r:r['seq'])
            unknown = db.execute('SELECT COUNT(*) FROM events WHERE turn=? AND at_ms IS NULL',(turn,)).fetchone()[0] if turn else 0
            oldest = db.execute('SELECT MIN(seq) FROM events WHERE turn=?',(turn,)).fetchone()[0] if turn else None
        records=[];used=0
        for row in (rows if after is not None else reversed(rows)):
            if used + len(row['payload'].encode()) > PAGE_BYTES and records:
                break
            used += len(row['payload'].encode())
            for item in json.loads(row['payload']):
                item['id']=str(row['seq']); item['event_seq']=row['seq'];item['turn']=row['turn']
                records.append((row['seq'],item))
        records.sort(key=lambda r:r[0])
        first=records[0][0] if records else None
        return {'items':merge_rows(records), 'turn':turn or '', 'events_recorded':bool(rows) or oldest is not None,
                'next_before':first if first and oldest is not None and first>oldest else None,
                'next_after':records[-1][0] if records else after, 'unknown_time_count':unknown,
                'association':'linked' if selected and selected.get('event_seq') else 'turn_only' if selected else None,
                'focus_event':selected.get('event_seq') if selected else None, 'cursor_kind':'sequence'}

    def source_view(self, snapshot_id, *, name=None, offset=0, limit=12000, across_revisions=False):
        snapshot = self.snapshot(snapshot_id)
        if not snapshot:
            raise ValueError('Unknown history snapshot')
        files = snapshot.get('files',{})
        names = sorted(n for n in files if Path(n).suffix.lower() in {'.py','.json','.yaml','.yml','.toml','.txt','.scad'})
        selected = name or ('model.py' if 'model.py' in names else next(iter(names),None))
        if selected and selected not in names:
            raise ValueError('File is not bound to this snapshot')
        previous=next(iter(self.preceding_snapshots(snapshot,limit=1,across_revisions=across_revisions)),None)
        offset=max(0,int(offset));limit=min(12000,max(1,int(limit)))
        def read(mapping):
            path=Path(mapping.get(selected,'')) if selected else None
            if not path or not path.is_file():return '',0
            size=path.stat().st_size
            with path.open('rb') as stream:
                stream.seek(offset);raw=stream.read(limit)
            return raw.decode('utf-8',errors='replace'),size
        content,size=read(files)
        old,_=read(previous.get('files',{}) if previous else {})
        diff=''.join(difflib.unified_diff(old.splitlines(keepends=True),content.splitlines(keepends=True),fromfile='previous/'+str(selected),tofile=str(selected)))
        return {'snapshot':snapshot_id,'files':names,'file':selected,'content':display_text(content),'diff':display_text(diff),
                'offset':offset,'next_offset':offset+limit if offset+limit<size else None,'bytes':size,
                'comparison':previous['id'] if previous else None,'comparison_revision':previous['revision'] if previous else None,'partial':offset>0 or offset+limit<size,
                'association':snapshot.get('capture'),'source_complete':snapshot.get('source_complete',False)}
