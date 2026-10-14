const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../dashboard/static/app.js'), 'utf8');
function functionsBetween(start, end) {
  return source.slice(source.indexOf('  function '+start), source.indexOf('  function '+end));
}
function fixture() {
  let rowWrites=0, chipWrites=0, listeners=0;
  const buttons=['A','B'].map(id=>{
    const classes=new Set(id==='A' ? ['is-active'] : []), attrs=new Map();
    return {dataset:{studioProject:id},classList:{contains:x=>classes.has(x),toggle:(x,on)=>on?classes.add(x):classes.delete(x)},
      getAttribute:k=>attrs.get(k),hasAttribute:k=>attrs.has(k),setAttribute:(k,v)=>attrs.set(k,v),removeAttribute:k=>attrs.delete(k)};
  });
  const listener={addEventListener:()=>listeners++};
  const scroll={scrollTop:321,set innerHTML(v){rowWrites++;this.scrollTop=0;}};
  const chips={set innerHTML(v){chipWrites++;},parentElement:listener};
  const count={textContent:''};
  const rail={dataset:{},querySelector:s=>s==='.studio-project-scroll'?scroll:s==='.project-status-chips'?chips:s==='.studio-rail-count'?count:listener,
    querySelectorAll:()=>buttons};
  const state={runId:'A',projects:[{id:'A',title:'Alpha',status:'preparing'},{id:'B',title:'Beta',status:'preparing'}],summaries:{},projectStatusFilter:'all'};
  const c={state,app:{querySelector:()=>rail},filteredProjects:()=>({items:state.projects,counts:{all:2,preparing:2}}),
    studioProjectName:p=>p.title,teamLibraryTree:()=>'',projectForRun:id=>state.projects.find(p=>p.id===id),
    projectRailKey:()=>c.projectListKey(),projectButtons:()=>'<buttons>',projectStatusChips:()=>'<chips>',bindProjectButtons:()=>{},observeProjectCovers:()=>{},render:()=>{}};
  vm.createContext(c);
  vm.runInContext(functionsBetween('projectListKey(', 'projectButtons(')+functionsBetween('updateProjectFilter(', 'studioAssets(')+functionsBetween('bindProjectRailEvents(', 'bindStudioEvents('),c);
  rail.dataset.projectListKey=c.projectListKey();
  return {c,state,rail,buttons,scroll,stats:()=>({rowWrites,chipWrites,listeners})};
}
test('elapsed-time summary polls preserve rows and scroll; selected project updates independently',()=>{
  const f=fixture();
  for(let i=0;i<4;i++) { f.state.summaries.A={status:'preparing',elapsed_seconds:i,updated_at:i};f.c.updateProjectFilter(); }
  f.state.runId='B';f.c.updateProjectFilter();
  assert.equal(f.stats().rowWrites,0);assert.equal(f.scroll.scrollTop,321);
  assert.equal(f.buttons[0].classList.contains('is-active'),false);
  assert.equal(f.buttons[1].classList.contains('is-active'),true);
  assert.equal(f.buttons[1].getAttribute('aria-current'),'true');
});
test('visible changes update once and retain scroll, including an equivalent-membership status filter',()=>{
  const f=fixture();f.state.summaries.A={status:'failed'};f.c.updateProjectFilter();f.c.updateProjectFilter();
  assert.equal(f.stats().rowWrites,1);assert.equal(f.scroll.scrollTop,321);
  f.state.projectStatusFilter='preparing';f.c.updateProjectFilter();
  assert.equal(f.stats().chipWrites,2);
});
test('retained rail binds its filter controls only once',()=>{
  const f=fixture();f.c.bindProjectRailEvents(f.rail);f.c.bindProjectRailEvents(f.rail);
  assert.equal(f.stats().listeners,3);
});

test('source drawer without a captured frame does not interrupt project rendering',()=>{
  const start=source.indexOf('  function historySourcePanel()');
  const end=source.indexOf('\n  function ',start+5);
  const c={state:{historySource:null},designFrame:()=>null,esc:v=>String(v??'')};
  vm.createContext(c);vm.runInContext(source.slice(start,end),c);
  assert.doesNotThrow(()=>c.historySourcePanel());
  assert.ok(!c.historySourcePanel().includes('data-history-source-file'));
});
