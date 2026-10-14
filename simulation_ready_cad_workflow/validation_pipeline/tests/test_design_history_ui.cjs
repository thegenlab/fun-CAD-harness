const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../dashboard/static/viewer3d.source.js'), 'utf8');
const begin = source.indexOf('  async replaceModel(url)');
const end = source.indexOf('  async prefetchHistory(', begin);
const context = {};
vm.createContext(context);
vm.runInContext('this.Replacer = class {' + source.slice(begin, end) + '}', context);

test('selecting A again cancels a pending B model without changing the camera or visible model', async () => {
  const viewer = new context.Replacer();
  let finish;
  Object.assign(viewer, {mode:'design', modelSrc:'A', model:'shape A', modelSelection:0,
    container:{dataset:{}}, historyModel:()=>new Promise(resolve=>finish=resolve), trimHistory:()=>{},
    scene:{remove:()=>assert.fail('removed selected A'), add:()=>assert.fail('displayed stale B')}});
  const pending = viewer.replaceModel('B');
  await viewer.replaceModel('A');
  finish({root:'shape B', materials:[], styled:true});
  await pending;
  assert.equal(viewer.modelSrc, 'A');
  assert.equal(viewer.model, 'shape A');
});

test('history cache disposes evicted geometry and retains at most three models', () => {
  const start = source.indexOf('  trimHistory('), stop = source.indexOf('  async replaceModel(', start);
  const disposed=[];
  const c = {disposeTree:(root)=>disposed.push(root), Set};vm.createContext(c);
  vm.runInContext('this.Cache=class {'+source.slice(start,stop)+'}',c);
  const cache=new c.Cache();cache.modelSrc='C';cache.container={dataset:{}};
  cache.historyModels=new Map(['A','B','C','D','E'].map(root=>[root,{root,materials:[]} ]));
  cache.trimHistory(['D','E']);
  assert.equal(cache.historyModels.size,3);assert.deepEqual(disposed,['A','B']);
  assert.deepEqual([...cache.historyModels.keys()],['C','D','E']);
});

test('legacy cached timelines remain available when no new history has been indexed', () => {
  const app = fs.readFileSync(path.join(__dirname, '../dashboard/static/app.js'), 'utf8');
  const start = app.indexOf('  function designTimeline()'), stop = app.indexOf('  function designFrame()',start);
  const c = {state:{runId:'run',historyRun:'run',history:{total:0,items:[]},data:{studio_preview:{timeline:['old1','old2']}}}};
  vm.createContext(c);vm.runInContext(app.slice(start,stop),c);
  assert.deepEqual(c.designTimeline(),['old1','old2']);
});

test('team project folders link to the latest snapshot in that folder', () => {
  const app = fs.readFileSync(path.join(__dirname, '../dashboard/static/app.js'), 'utf8');
  assert.match(app, /class="team-project"/);
  assert.match(app, /snapshots\[snapshots\.length - 1\]\.url/);
  assert.match(app, /a\.team-snapshot, a\.team-project, a\.team-library-home/);
});

test('history playback includes 0.1s and 0.25s intervals', () => {
  const app = fs.readFileSync(path.join(__dirname, '../dashboard/static/app.js'), 'utf8');
  assert.match(app, /\[100,250,500,1000,2000\]/);
});

test('a new task can render before its first history record exists', () => {
  const app = fs.readFileSync(path.join(__dirname, '../dashboard/static/app.js'), 'utf8');
  const start = app.indexOf('  function designTimeline()'), stop = app.indexOf('  function designFrameIndex()',start);
  const c={state:{runId:'new',history:null,data:{studio_preview:{timeline:[]}}}};
  vm.createContext(c);vm.runInContext(app.slice(start,stop),c);
  assert.equal(c.displayDesignFrame(),null);
});
