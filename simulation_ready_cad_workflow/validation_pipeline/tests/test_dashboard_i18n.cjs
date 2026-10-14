const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),vm=require('node:vm');
const staticRoot=path.join(__dirname,'../dashboard/static');
const runtime=fs.readFileSync(path.join(staticRoot,'i18n.js'),'utf8');
const catalogSource=fs.readFileSync(path.join(staticRoot,'i18n/en.js'),'utf8');

// Minimal DOM: enough for the tree walk, attribute and mutation handling.
class FakeNode {
  constructor(type){this.nodeType=type;this.parentNode=null;this.childNodes=[];}
  appendChild(child){child.parentNode=this;this.childNodes.push(child);return child;}
}
class FakeText extends FakeNode { constructor(value){super(3);this.nodeValue=value;} }
class FakeElement extends FakeNode {
  constructor(name,attrs={}){super(1);this.nodeName=name.toUpperCase();this.attrs={...attrs};this.lang='';}
  getAttribute(name){return Object.prototype.hasOwnProperty.call(this.attrs,name) ? this.attrs[name] : null;}
  setAttribute(name,value){this.attrs[name]=String(value);}
  hasAttribute(name){return Object.prototype.hasOwnProperty.call(this.attrs,name);}
  closest(selector){
    const name=selector.replace(/^\[|\]$/g,'');
    for(let node=this;node&&node.nodeType===1;node=node.parentNode) if(node.hasAttribute(name)) return node;
    return null;
  }
}
function el(name,attrs,...children){
  const node=new FakeElement(name,attrs);
  for(const child of children) node.appendChild(typeof child==='string' ? new FakeText(child) : child);
  return node;
}
function text(node){return node.nodeType===3 ? node.nodeValue : node.childNodes.map(text).join('');}

function load({language='en-US',search='',storage={},cookie='',catalog=false,currentScript='http://host/base/static/i18n.js'}={}){
  const listeners=[],appended=[],observers=[],calls=[];
  const store={...storage};
  const html=el('html',{},el('head',{},el('title',{},'机械工作台')),el('body',{},el('p',{},'正在连接')));
  const document={
    cookie,
    currentScript:currentScript ? {src:currentScript} : null,
    documentElement:html,
    head:html.childNodes[0],
    createElement:(name)=>el(name),
    addEventListener:(type,fn)=>listeners.push([type,fn]),
  };
  document.head.appendChild=(node)=>{appended.push(node);return node;};
  class MutationObserver {
    constructor(callback){this.callback=callback;this.taken=0;observers.push(this);}
    observe(target,options){this.target=target;this.options=options;}
    takeRecords(){this.taken+=1;return [];}
  }
  const context={
    document,MutationObserver,URL,URLSearchParams,
    navigator:{language},
    localStorage:{getItem:(key)=>key in store ? store[key] : null,setItem:(key,value)=>{store[key]=String(value);}},
    location:{search,href:`http://host/base/${search}`,reload:()=>calls.push(['reload']),replace:(url)=>calls.push(['replace',url])},
  };
  vm.createContext(context);
  if(catalog) vm.runInContext(catalogSource,context);
  vm.runInContext(runtime,context);
  return {context,document,html,store,listeners,appended,observers,calls};
}

const sample={
  '通过':'Passed',
  '画面未变':'Frame unchanged',
  '继续处理：{0}':'Continue: {0}',
  '本版本未通过':'This revision failed',
  '{0} 个版本 Token 待结算':'{0} revisions with tokens pending',
  '{0}时{1}分{2}':'{0}h {1}m {2}',
  '{0}秒':'{0}s',
  '实验性':'experimental',
  '查看 {0} 详情':'View {0} details',
  '正在连接':'Connecting',
  '机械工作台':'Mechanism Workbench',
  '搜索':'Search',
  '开始':'Start',
  '{0}分{1}':'{0}m {1}',
  '{0} 分钟前':'{0} min ago',
  '编辑 {0}':'Edit {0}',
  '全部':'All',
};

test('exact, template and segment lookups keep surrounding whitespace and punctuation',()=>{
  const {context}=load();
  const tr=context.i18n.createTranslator(sample);
  assert.equal(tr.translate('  通过\n'),'  Passed\n');
  assert.equal(tr.translate('继续处理：本版本未通过。'),'Continue: This revision failed.');
  assert.equal(tr.translate('3 个版本 Token 待结算'),'3 revisions with tokens pending');
  assert.equal(tr.translate('2时5分30秒'),'2h 5m 30s');
  assert.equal(tr.translate('通过 · Step 3/5 · 画面未变'),'Passed · Step 3/5 · Frame unchanged');
  assert.equal(tr.translate('Cursor（实验性）'),'Cursor (experimental)');
  assert.equal(tr.translate('未登记的文字'),'未登记的文字');
  assert.equal(tr.translate('5 分钟前：编辑 foo.py'),'5 min ago: Edit foo.py');
  assert.equal(tr.translate('全部 3'),'All 3');
  // A template with little fixed text never leaves Chinese in its captures.
  assert.equal(tr.translate('5分钟'),'5分钟');
  assert.equal(tr.translate('plain English'),'plain English');
  assert.equal(tr.translate(null),null);
});

test('the tree walk skips code, opted-out content and non-button values',()=>{
  const {context}=load();
  const tr=context.i18n.createTranslator(sample);
  const root=el('div',{title:'查看 r01 详情'},
    el('span',{},'通过'),
    el('pre',{},'通过'),
    el('code',{},'通过'),
    el('textarea',{placeholder:'搜索'},'通过'),
    el('p',{translate:'no','aria-label':'通过'},'通过',el('b',{},'通过')),
    el('div',{'data-no-i18n':''},'通过'),
    el('input',{type:'submit',value:'开始'}),
    el('input',{type:'text',value:'开始',placeholder:'搜索'}),
  );
  tr.translateTree(root);
  const [span,pre,code,area,optedOut,noI18n,submit,input]=root.childNodes;
  assert.equal(root.getAttribute('title'),'View r01 details');
  assert.equal(text(span),'Passed');
  assert.equal(text(pre),'通过');
  assert.equal(text(code),'通过');
  assert.equal(text(area),'通过');
  assert.equal(area.getAttribute('placeholder'),'Search');
  assert.equal(text(optedOut),'通过通过');
  assert.equal(optedOut.getAttribute('aria-label'),'通过');
  assert.equal(text(noI18n),'通过');
  assert.equal(submit.getAttribute('value'),'Start');
  assert.equal(input.getAttribute('value'),'开始');
  assert.equal(input.getAttribute('placeholder'),'Search');
});

test('mutations inside opted-out containers are ignored',()=>{
  const {context}=load();
  const tr=context.i18n.createTranslator(sample);
  const agent=el('div',{translate:'no'});
  const ui=el('div',{});
  const root=el('main',{},agent,ui);
  const message=agent.appendChild(el('p',{},'通过'));
  const label=ui.appendChild(el('p',{},'通过'));
  const changed=ui.appendChild(new FakeText('画面未变'));
  const titled=ui.appendChild(el('button',{title:'通过'}));
  tr.applyMutations([
    {type:'childList',addedNodes:[message,label]},
    {type:'characterData',target:changed},
    {type:'attributes',target:titled,attributeName:'title'},
  ]);
  assert.equal(text(message),'通过');
  assert.equal(text(label),'Passed');
  assert.equal(changed.nodeValue,'Frame unchanged');
  assert.equal(titled.getAttribute('title'),'Passed');
  assert.equal(root.childNodes.length,2);
});

test('Chinese mode loads no catalog and observes nothing',()=>{
  const {context,appended,observers,html}=load({language:'zh-CN'});
  assert.equal(context.i18n.lang,'zh');
  assert.equal(context.t('通过'),'通过');
  assert.equal(appended.length,0);
  assert.equal(observers.length,0);
  assert.equal(text(html),'机械工作台正在连接');
  assert.match(context.i18n.toggleMarkup(),/data-lang-toggle translate="no"/);
});

test('English mode fetches the catalog next to i18n.js and translates once it arrives',()=>{
  const {context,appended,observers,html}=load({language:'fr-FR'});
  assert.equal(context.i18n.lang,'en');
  assert.equal(html.lang,'en');
  assert.equal(appended.length,1);
  assert.equal(appended[0].src,'http://host/base/static/i18n/en.js');
  assert.equal(text(html),'机械工作台正在连接');
  context.QUEST_I18N_CATALOGS.en=sample;
  context.QUEST_I18N_CATALOGS.onload('en');
  assert.equal(text(html),'Mechanism WorkbenchConnecting');
  assert.equal(context.t('本版本未通过'),'This revision failed');
  assert.equal(observers.length,1);
  assert.equal(observers[0].target,context.document);
  // Records produced by the observer's own writes are discarded.
  const added=html.childNodes[1].appendChild(el('p',{},'通过'));
  observers[0].callback([{type:'childList',addedNodes:[added]}]);
  assert.equal(text(added),'Passed');
  assert.equal(observers[0].taken,1);
});

test('?lang= overrides and persists; stored choice beats the browser language',()=>{
  const fromQuery=load({language:'zh-CN',search:'?lang=en'});
  assert.equal(fromQuery.context.i18n.lang,'en');
  assert.equal(fromQuery.store['quest-dashboard-lang'],'en');
  assert.match(fromQuery.document.cookie,/^lang=en;/);
  const stored=load({language:'en-US',storage:{'quest-dashboard-lang':'zh'}});
  assert.equal(stored.context.i18n.lang,'zh');
  const cookie=load({language:'en-US',cookie:'other=1; lang=zh'});
  assert.equal(cookie.context.i18n.lang,'zh');
});

test('the toggle stores the other language and reloads without the ?lang override',()=>{
  const zh=load({language:'zh-CN'});
  const [type,onClick]=zh.listeners[0];
  assert.equal(type,'click');
  const button=el('button',{'data-lang-toggle':''},el('span',{},'English'));
  onClick({target:button.childNodes[0],preventDefault(){}});
  assert.equal(zh.store['quest-dashboard-lang'],'en');
  assert.deepEqual(zh.calls,[['reload']]);
  const en=load({language:'zh-CN',search:'?lang=en&run=r01'});
  en.context.i18n.setLang('zh');
  assert.deepEqual(en.calls,[['replace','http://host/base/?run=r01']]);
});

test('the real catalog translates text rendered by the dashboard',()=>{
  const {context,html}=load({language:'en-GB',catalog:true});
  assert.equal(text(html),'Mechanism WorkbenchConnecting');
  const t=context.t;
  assert.equal(t('新任务'),'New task');
  assert.equal(t('▶ 播放 3 个画面'),'▶ '+t('播放 {0} 个画面').replace('{0}','3'));
  assert.doesNotMatch(t('通过 · Step 3/5 · 画面未变'),/[一-鿿]/);
  assert.doesNotMatch(t('缺少 Claude Code 沙箱工具：bwrap, socat'),/[一-鿿]/);
  assert.doesNotMatch(t('停止这个运行？Harness 会保存已有进度并写出停止摘要，之后可以继续运行。'),/[一-鿿]/);
});
