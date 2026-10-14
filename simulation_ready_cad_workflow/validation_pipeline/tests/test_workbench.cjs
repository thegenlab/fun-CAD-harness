const {test}=require('node:test');
const assert=require('node:assert/strict');
const wb=require('../dashboard/static/workbench.js');

test('filters names and IDs without mutating the project collection',()=>{
 const projects=[{id:'pump_01',title:'Pump',status:'running'},{id:'pump_02',title:'Other',status:'failed'},{id:'crank',title:'Crank',status:'passed'}];
 const result=wb.filterProjects(projects,'pump','failed');
 assert.deepEqual(result.items.map(x=>x.id),['pump_02']);assert.equal(result.counts.all,2);assert.equal(result.counts.running,1);assert.equal(projects.length,3);
});
test('notification baseline and dedup exclude old results and intermediate failures',()=>{
 const tracker=new wb.Notices();
 const row=(status,key,notify)=>({id:'p',run_id:'r1',status,notice_key:key,notify});
 assert.deepEqual(tracker.update({ready:true,projects:[row('blocked','old',true)]}),[]);
 assert.deepEqual(tracker.update({ready:true,projects:[row('failed','repairing',false)]}),[]);
 assert.equal(tracker.update({ready:true,projects:[row('passed','done',true)]}).length,1);
 assert.equal(tracker.update({ready:true,projects:[row('passed','done',true)]}).length,0);
 const reload=new wb.Notices(tracker.seen);assert.equal(reload.update({ready:true,projects:[row('passed','done',true)]}).length,0);
});
test('search includes loaded tool output and title belongs to current project',()=>{
 const items=[{title:'构建',text:'run build',output:'error in model.py'},{title:'消息',text:'ready'}];
 assert.equal(wb.searchEvents(items,'model.py error').length,1);assert.equal(wb.searchEvents(items,'missing').length,0);
 assert.match(wb.title({title:'A',status:'running'},2),/↻.*2.*A/);
});
