const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../dashboard/static/app.js'), 'utf8');
const start = source.indexOf('  async function refreshTrajectory()');
const end = source.indexOf('  function partsPanel()', start);

test('pending refresh cannot overwrite a newly selected turn or older page', async () => {
  for (const selection of [{ trajectoryTurn: 't001_author' }, { trajectoryTurn: 't002_author', trajectoryBefore: 100 }]) {
    const requests = [];
    const state = { studioDrawer: 'activity', data: { live_workflow: {} }, runId: 'run', trajectoryRun: 'run',
      trajectoryTurn: '', trajectoryBefore: null, trajectoryLoading: false, trajectory: null };
    const context = { state, boot: { mode: 'live', apiBase: '/api' }, URLSearchParams,
      app: { querySelector: () => null }, trajectoryExpanded: new Set(),
      fetch: (url) => new Promise(resolve => requests.push({ url, resolve })) };
    vm.createContext(context);
    vm.runInContext(source.slice(start, end), context);
    const first = context.refreshTrajectory();
    Object.assign(state, selection);
    await context.refreshTrajectory();
    assert.equal(requests.length, 1);
    requests[0].resolve({ ok: true, json: async () => ({ turn: 'stale', items: [] }) });
    await first;
    assert.equal(state.trajectory, null);
    assert.equal(requests.length, 2);
    assert.ok(requests[1].url.includes('turn=' + selection.trajectoryTurn));
    if (selection.trajectoryBefore) assert.ok(requests[1].url.includes('before=100'));
    requests[1].resolve({ ok: true, json: async () => ({ turn: selection.trajectoryTurn, items: ['correct'] }) });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(state.trajectory.turn, selection.trajectoryTurn);
    assert.equal(state.trajectory.items[0], 'correct');
  }
});

test('test outcome labels distinguish unrun, timeout, error and completed validation', () => {
  const start = source.indexOf('  function studioTestOutcome(');
  const end = source.indexOf('  function partsPanel()', start);
  const context = {}; vm.createContext(context); vm.runInContext(source.slice(start, end), context);
  assert.equal(context.studioTestOutcome({execution_started:false}), '未运行');
  assert.equal(context.studioTestOutcome({execution_status:'SIMULATION_TIMEOUT'}), '超时');
  assert.equal(context.studioTestOutcome({failure_status:'HARNESS_ERROR'}), '运行错误');
  assert.equal(context.studioTestOutcome({physical_pass:true,geometry_audit:{passed:true}}), '通过');
  assert.equal(context.studioTestOutcome({physical_pass:false}), '失败');
});
