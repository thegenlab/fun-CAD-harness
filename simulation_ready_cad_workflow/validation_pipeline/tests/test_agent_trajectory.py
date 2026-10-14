"""Read-only event paging through the same endpoint used by the studio."""
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import quote
from unittest.mock import patch

from ..dashboard.agent_trajectory import trajectory_view
from .test_dashboard_studio_live import make_project
from .test_dashboard_changes import _dashboard, _get


class AgentTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.run_id = make_project(self.root)
        self.turn = self.run.parent.parent/'worker_turns'/'t001_author'
        self.turn.mkdir(parents=True)
        self.events = self.turn/'events.jsonl'

    def write_events(self, events):
        self.events.write_text(''.join(json.dumps(e)+'\n' for e in events), encoding='utf-8')

    def view(self, **kwargs):
        return trajectory_view(workspace_root=self.root, run_dir=self.run, **kwargs)

    def test_public_message_and_tool_lifecycle_with_redaction(self):
        self.write_events([
            {'type':'item.completed','item':{'id':'private','type':'reasoning','text':'PRIVATE_ANALYSIS'}},
            {'type':'item.completed','item':{'id':'message','type':'agent_message','text':'Inspect the mechanism.'}},
            {'type':'item.started','item':{'id':'tool','type':'command_execution','command':'echo OPENAI_API_KEY=canary-secret','status':'in_progress'}},
            {'type':'item.completed','item':{'id':'tool','type':'command_execution','command':'echo OPENAI_API_KEY=canary-secret','status':'completed','exit_code':0,'aggregated_output':'Authorization: Bearer canary-bearer'}},
        ])
        (self.turn/'prompt.txt').write_text('PRIVATE_SYSTEM_PROMPT')
        before = self.events.read_bytes()
        value = self.view()
        self.assertEqual(['message','tool'], [row['kind'] for row in value['items']])
        self.assertEqual('completed', value['items'][1]['status'])
        self.assertEqual(0, value['items'][1]['exit_code'])
        encoded = json.dumps(value)
        for secret in ('PRIVATE_ANALYSIS','PRIVATE_SYSTEM_PROMPT','canary-secret','canary-bearer'):
            self.assertNotIn(secret, encoded)
        self.assertEqual(before,self.events.read_bytes())

    def test_pages_keep_chronology_and_live_partial_tail_is_retried(self):
        self.write_events([{'type':'item.completed','item':{'id':str(i),'type':'agent_message','text':str(i)}} for i in range(150)])
        with self.events.open('a') as stream:stream.write('{"type":')
        latest = self.view()
        self.assertEqual([str(i) for i in range(50,150)], [row['text'] for row in latest['items']])
        older = self.view(before=latest['next_before'])
        self.assertEqual([str(i) for i in range(50)], [row['text'] for row in older['items']])
        self.assertIsNone(older['next_before'])
        with self.events.open('a') as stream:stream.write('"item.completed","item":{"id":"live","type":"agent_message","text":"new"}}\n')
        self.assertEqual('new',self.view()['items'][-1]['text'])

    def test_app_notifications_project_public_lifecycle_and_preserve_raw_logs(self):
        def notification(method, item, stamp=2000):
            return {'method':method, 'params':{'threadId':'thread', 'turnId':'turn', 'item':item}, 'emittedAtMs':stamp}
        self.write_events([
            notification('item/completed', {'id':'private','type':'reasoning','text':'PRIVATE_ANALYSIS'}),
            {'method':'account/rateLimits/updated','params':{'private':'PRIVATE_ACCOUNT'},'emittedAtMs':2000},
            notification('item/completed', {'id':'message','type':'agentMessage','text':'Inspect the mechanism.'}),
            notification('item/started', {'id':'command','type':'commandExecution','command':'echo ready','status':'inProgress'}),
            notification('item/completed', {'id':'command','type':'commandExecution','command':'echo ready',
                         'status':'completed','exitCode':0,'aggregatedOutput':'ready'}),
            notification('item/started', {'id':'mcp','type':'mcpToolCall','tool':'inspect','arguments':{},'status':'inProgress'}),
            notification('item/completed', {'id':'mcp','type':'mcpToolCall','tool':'inspect','arguments':{},
                         'status':'completed','result':{'attempt':'/workspace/.runtime_runs/author_tools/build-app'}}),
            notification('item/completed', {'id':'file','type':'fileChange','status':'completed',
                         'changes':[{'path':'task.py','diff':'+def build(): pass'}]}),
            {'method':'turn/completed','params':{'turn':{'id':'turn','status':'completed'}},'emittedAtMs':2500},
        ])
        before = self.events.read_bytes()
        from ..trajectory.store import HistoryStore
        store = HistoryStore(self.run.parent.parent)
        self.assertEqual(store.index_log(self.events, self.turn.name, drain=True), 7)
        self.assertTrue(store.tool_completed)
        value = store.event_page(turn=self.turn.name)
        self.assertEqual([row['kind'] for row in value['items']], ['message','tool','tool','tool','status'])
        self.assertEqual(value['items'][1]['output'], 'ready')
        self.assertEqual(value['items'][1]['exit_code'], 0)
        self.assertEqual(value['items'][1]['status'], 'completed')
        self.assertEqual(value['unknown_time_count'], 0)
        self.assertEqual(len(store.event_page(turn=self.turn.name, from_ms=2400)['items']), 1)
        with store.connect() as db:
            association = db.execute('SELECT turn,at_ms FROM attempt_events WHERE id=?', ('build-app',)).fetchone()
        self.assertEqual(tuple(association), (self.turn.name, 2000))
        encoded = json.dumps(value)
        self.assertNotIn('PRIVATE_ANALYSIS', encoded)
        self.assertNotIn('PRIVATE_ACCOUNT', encoded)
        self.assertEqual(self.events.read_bytes(), before)

    def test_app_normalized_command_retains_camelcase_output(self):
        from ..trajectory.events import event_rows, completed_tool_output
        event = {'type':'item.completed','item':{'id':'command','type':'command_execution',
                 'command':'echo ready','aggregatedOutput':'ready','exitCode':0,'status':'completed'}}
        self.assertEqual(event_rows(event, 0)[0]['output'], 'ready')
        self.assertEqual(completed_tool_output(event), 'ready')

    def test_cursor_tools_and_missing_event_history(self):
        self.write_events([
            {'type':'assistant','message':{'content':[{'type':'text','text':'Build the model.'}]}},
            {'type':'tool_call','subtype':'started','call_id':'c1','tool_call':{'shellToolCall':{'args':{'command':'python model.py'}}}},
            {'type':'tool_call','subtype':'completed','call_id':'c1','tool_call':{'shellToolCall':{'args':{'command':'python model.py'},'result':{'output':'built'}}}},
        ])
        self.assertEqual(2,len(self.view()['items']))
        self.assertIn('built',self.view()['items'][-1]['output'])
        self.events.unlink()
        (self.turn/'response.json').write_text(json.dumps({'status':'ready','summary':'Model delivered.'}))
        value=self.view()
        self.assertFalse(value['events_recorded'])
        self.assertEqual('Model delivered.',value['response']['summary'])

    def test_large_tool_record_is_visible_and_pagination_moves_back(self):
        self.write_events([{'type':'item.completed','item':{'type':'command_execution','command':'large-output',
            'aggregated_output':'x'*(800*1024),'status':'completed','exit_code':0}}])
        value=self.view()
        self.assertEqual('长记录未展开',value['items'][0]['title'])
        self.assertGreater(value['next_before'],0)
        older=self.view(before=value['next_before'])
        self.assertIsNone(older['next_before'])

    def test_pending_feedback_is_retained_without_empty_issue_noise(self):
        (self.run/'result.json').write_text(json.dumps({'status':'ACTIVE_BUDGET_EXHAUSTED','issues':[],
            'pending_tests':['loaded_motion']}))
        feedback=self.view()['feedback']
        self.assertNotIn('issues',feedback)
        self.assertIn('loaded_motion',feedback['pending_tests'])

    def test_time_window_filters_events_by_millisecond_timestamp(self):
        self.write_events([
            {'type':'assistant','message':{'content':[{'type':'text','text':'early'}]},'timestamp_ms':1000},
            {'type':'assistant','message':{'content':[{'type':'text','text':'mid'}]},'timestamp_ms':2000},
            {'type':'assistant','message':{'content':[{'type':'text','text':'late'}]},'timestamp_ms':3000},
            {'type':'assistant','message':{'content':[{'type':'text','text':'no-stamp'}]}},
            {'type':'item.completed','timestamp':'2026-09-12T03:48:25+00:00',
             'item':{'id':'codex','type':'agent_message','text':'iso-stamp'}},
        ])
        from ..trajectory.store import HistoryStore
        HistoryStore(self.run.parent.parent).index_log(self.events, self.turn.name, drain=True)
        value = self.view(from_ms=1500, to_ms=2500)
        self.assertEqual(['mid'], [row['text'] for row in value['items']])
        self.assertEqual({'from_ms':1500,'to_ms':2500}, value['time_filter'])
        open_end = self.view(from_ms=2500)
        self.assertEqual(['late','iso-stamp'], [row['text'] for row in open_end['items']])
        from datetime import datetime, timezone
        iso_ms = int(datetime(2026, 9, 12, 3, 48, 25, tzinfo=timezone.utc).timestamp() * 1000)
        closed = self.view(from_ms=1000, to_ms=iso_ms)
        self.assertNotIn('iso-stamp', [row['text'] for row in closed['items']])
        self.assertNotIn('no-stamp', [row['text'] for row in self.view(from_ms=0)['items']])

    def test_endpoint_uses_only_selected_project_and_rejects_unknown_turn(self):
        self.write_events([{'type':'item.completed','item':{'type':'agent_message','text':'selected'}}])
        other,_=make_project(self.root,'another')
        outside=other.parent.parent/'worker_turns'/'t099_author';outside.mkdir(parents=True)
        (outside/'response.json').write_text(json.dumps({'summary':'OTHER_PROJECT'}))
        with _dashboard(self.root) as server, patch.object(server,'studio_preview',side_effect=AssertionError('Trajectory must not render')):
            status,body,_=_get(server,'/api/agent-trajectory?run='+quote(self.run_id,safe=''))
            self.assertEqual(200,status)
            self.assertNotIn(b'OTHER_PROJECT',body)
            status,_,_=_get(server,'/api/agent-trajectory?run='+quote(self.run_id,safe='')+'&turn=t099_author')
            self.assertEqual(404,status)
            status,_,_=_get(server,'/api/agent-trajectory?run='+quote(self.run_id,safe='')+'&turn=../../outside')
            self.assertEqual(404,status)


if __name__ == '__main__': unittest.main()
