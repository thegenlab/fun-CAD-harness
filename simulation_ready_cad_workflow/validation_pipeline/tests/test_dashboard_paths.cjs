const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../dashboard/static/app.js'),'utf8');
test('worker recovery submits under the configured dashboard prefix',async()=>{
 let submit,request;const attrs={};const button={disabled:false};
 const form={dataset:{},addEventListener:(type,handler)=>{submit=handler;},setAttribute:(key,value)=>attrs[key]=value,querySelector:()=>button};
 const elements={'worker-recovery-form':form,'worker-recovery-actor':{value:'local operator'},'worker-recovery-note':{value:'continue'},'worker-recovery-message':{}};
 const context={document:{getElementById:id=>elements[id]},boot:{apiBase:'/harness/dashboard/api'},state:{runId:'project/r01'},humanDecisionCapability:'test-capability',
 setResourceReviewMessage(){},renderWorkerRecoverySlot(){},toast(){},fetch:async(url,options)=>{request={url,options};return {ok:true,json:async()=>({event_kernel:{status:'QUEUED'}})};}};
 vm.createContext(context);const start=source.indexOf('  function bindWorkerRecoveryEvents()');const end=source.indexOf('\n  function ',start+5);
 vm.runInContext(source.slice(start,end),context);context.bindWorkerRecoveryEvents();await submit({preventDefault(){}});
 assert.equal(request.url,'/harness/dashboard/api/worker-resume');assert.equal(request.options.method,'POST');assert.equal(JSON.parse(request.options.body).run,'project/r01');
});
test('workspaceFromUrl maps import snapshots and the local catalog',()=>{
 const context={URL};vm.createContext(context);
 const start=source.indexOf('  function workspaceFromUrl(');
 const end=source.indexOf('\n  function ',start+5);
 vm.runInContext(source.slice(start,end),context);
 const options={origin:'https://example.org/harness/dashboard/',localUrl:'/harness/dashboard/',currentBasePath:'/harness/dashboard'};
 const imported=context.workspaceFromUrl('/harness/dashboard/imports/abc/',options);
 assert.equal(imported.basePath,'/harness/dashboard/imports/abc');
 assert.equal(imported.apiBase,'/harness/dashboard/imports/abc/api');
 const queried=context.workspaceFromUrl('https://example.org/harness/dashboard/imports/abc/?run=old',options);
 assert.equal(queried.basePath,'/harness/dashboard/imports/abc');
 const local=context.workspaceFromUrl('/harness/dashboard/',{...options,currentBasePath:'/harness/dashboard/imports/abc'});
 assert.equal(local.basePath,'/harness/dashboard');
 assert.equal(local.apiBase,'/harness/dashboard/api');
 assert.equal(context.workspaceFromUrl('/harness/dashboard/maintenance',options),null);
 assert.equal(context.workspaceFromUrl('/other/app/',options),null);
});
