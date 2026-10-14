const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../dashboard/static/launch.js'),'utf8');
function load(name){
 const context={};vm.createContext(context);
 const start=source.indexOf(`  function ${name}(`);const end=source.indexOf('\n  function ',start+5);
 vm.runInContext(source.slice(start,end),context);return context[name];
}
test('a launched run opens the live project view and keeps the capability hash',()=>{
 const url=load('launchRedirectUrl');
 assert.equal(url('','.runtime_runs/projects/demo/runs/r01','#human=abc'),'/?run=.runtime_runs%2Fprojects%2Fdemo%2Fruns%2Fr01&view=current&refresh=1#human=abc');
 assert.equal(url('/harness/dashboard','.runtime_runs/projects/a b/runs/r01',''),'/harness/dashboard/?run=.runtime_runs%2Fprojects%2Fa%20b%2Fruns%2Fr01&view=current&refresh=1');
 assert.equal(url(undefined,'.runtime_runs/projects/x/runs/r01'),'/?run=.runtime_runs%2Fprojects%2Fx%2Fruns%2Fr01&view=current&refresh=1');
});
