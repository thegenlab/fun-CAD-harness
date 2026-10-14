import * as THREE from 'three';

const savedProjects = new Map();
const materials = mesh => Array.isArray(mesh.material) ? mesh.material : [mesh.material];
function nodePath(node, root) {
  const names=[];
  for(let current=node;current && current!==root;current=current.parent) {
    const siblings=current.parent?.children.filter(item=>item.name===current.name) || [];
    names.unshift((current.name || current.type)+(siblings.length>1 ? `[${siblings.indexOf(current)+1}]` : ''));
  }
  return names.join('/');
}

export class PartsController {
  constructor(viewer) {
    this.viewer=viewer;this.entries=[];this.explosion=0;
    this.project=(viewer.container.closest('[data-studio-run]')?.dataset.studioRun || '').split('/runs/')[0];
    const saved=savedProjects.get(this.project);
    this.hidden=new Set(saved?.hidden || []);this.selected=saved?.selected || '';
    let start;
    this.down=event=>{start={x:event.clientX,y:event.clientY};};
    this.up=event=>{
      if(!start || event.button!==0 || Math.hypot(event.clientX-start.x,event.clientY-start.y)>5)return;
      const box=viewer.renderer.domElement.getBoundingClientRect();
      const pointer=new THREE.Vector2((event.clientX-box.left)/box.width*2-1,-(event.clientY-box.top)/box.height*2+1);
      const ray=new THREE.Raycaster();ray.setFromCamera(pointer,viewer.camera);
      const visible=this.entries.filter(entry=>entry.mesh.visible && this.ancestorsVisible(entry.mesh));
      const hit=ray.intersectObjects(visible.map(entry=>entry.mesh),false)[0];
      this.action('select',this.entries.find(entry=>entry.mesh===hit?.object)?.id || '');
    };
    viewer.renderer.domElement.addEventListener('pointerdown',this.down);
    viewer.renderer.domElement.addEventListener('pointerup',this.up);
  }
  ancestorsVisible(mesh) {for(let parent=mesh.parent;parent;parent=parent.parent)if(!parent.visible)return false;return true;}
  attach(root) {
    root.updateWorldMatrix(true,true);this.explosion=0;
    this.entries=[];
    root.traverse(mesh=>{
      if(!mesh.userData.questPartSource)return;
      const component=mesh.userData.physicsComponent || {};
      const body=component.bodyId || mesh.userData.body_id || mesh.geometry?.userData?.body_id;
      const solid=component.solid_id || mesh.userData.source_solid_id || mesh.geometry?.userData?.source_solid_id;
      const path=nodePath(mesh,root);
      this.entries.push({id:body && solid ? JSON.stringify([body,solid]) : path,body:body || '',label:solid || mesh.name || '未命名实体',path,mesh,
        position:mesh.position.clone(),matrix:mesh.matrix.clone(),world:mesh.getWorldPosition(new THREE.Vector3()),visible:mesh.visible,
        colors:materials(mesh).map(material=>({emissive:material.emissive?.clone(),intensity:material.emissiveIntensity})),
        center:new THREE.Box3().setFromObject(mesh).getCenter(new THREE.Vector3())});
    });
    const counts=new Map();for(const entry of this.entries)counts.set(entry.id,(counts.get(entry.id)||0)+1);
    for(const entry of this.entries)if(counts.get(entry.id)>1)entry.id=entry.path;
    const available=new Set(this.entries.map(entry=>entry.id));
    this.hidden=new Set([...this.hidden].filter(id=>available.has(id)));
    if(!available.has(this.selected))this.selected='';
    if(this.viewer.physicsState)this.viewer.physicsState.part=this.entries.find(entry=>entry.id===this.selected)?.mesh.name || this.viewer.physicsState.part;
    const bounds=new THREE.Box3().setFromObject(root);this.center=bounds.getCenter(new THREE.Vector3());this.spread=bounds.getSize(new THREE.Vector3()).length()*.45;
    const groups=new Map();for(const entry of this.entries){const key=entry.body || entry.path;let box=groups.get(key);if(!box){box=new THREE.Box3();groups.set(key,box);}box.expandByObject(entry.mesh);}
    let index=0;for(const [key,box] of groups){let direction=box.getCenter(new THREE.Vector3()).sub(this.center);if(direction.lengthSq()<1e-12)direction.set(Math.cos(index*2.4),Math.sin(index*2.4),.5);direction.normalize().multiplyScalar(this.spread);for(const entry of this.entries)if((entry.body || entry.path)===key)entry.offset=direction;index++;}
    this.refresh();
  }
  detach() {
    for(const entry of this.entries){entry.mesh.position.copy(entry.position);entry.mesh.matrix.copy(entry.matrix);entry.mesh.visible=entry.visible;materials(entry.mesh).forEach((material,index)=>{if(entry.colors[index].emissive)material.emissive.copy(entry.colors[index].emissive);material.emissiveIntensity=entry.colors[index].intensity;});}
    this.explosion=0;this.entries=[];
  }
  apply() {
    const viewer=this.viewer;
    if(viewer.physicsState)this.selected=this.entries.find(entry=>entry.mesh.name===viewer.physicsState.part)?.id || '';
    for(const entry of this.entries){
      entry.mesh.visible=(viewer.mode==='physics' ? viewer.bodyIsVisible(entry.mesh.userData.physicsComponent?.bodyId) : entry.visible) && !this.hidden.has(entry.id);
      if(viewer.mode!=='physics')materials(entry.mesh).forEach((material,index)=>{if(!material.emissive)return;material.emissive.copy(entry.id===this.selected ? new THREE.Color(0x086e77) : entry.colors[index].emissive);material.emissiveIntensity=entry.id===this.selected ? .7 : entry.colors[index].intensity;});
    }
  }
  refresh() {
    if(this.viewer.mode==='physics')this.viewer.applyPhysicsFrame();else{this.apply();this.viewer.renderOnce();}
    this.publish();
  }
  action(action,id,value) {
    const entry=this.entries.find(item=>item.id===id);
    if(action==='select'){
      this.selected=entry?.id || '';
      if(this.viewer.physicsState)this.viewer.physicsState.part=entry?.mesh.name || '';
    } else if(action==='visible' && entry){
      if(value)this.hidden.delete(id);else this.hidden.add(id);
      if(value && this.viewer.mode==='physics')this.viewer.hiddenBodyIds.delete(entry.mesh.userData.physicsComponent?.bodyId);
    } else if(action==='isolate' && entry){this.hidden=new Set(this.entries.filter(item=>item.id!==id).map(item=>item.id));this.viewer.hiddenBodyIds?.clear();}
    else if(action==='all'){this.hidden.clear();this.viewer.hiddenBodyIds?.clear();}
    else if(action==='explode' && this.viewer.mode==='design'){
      this.explosion=Math.max(0,Math.min(100,Number(value)||0));
      // Rebuild each display transform from its saved assembly pose, in parent-first order.
      for(const item of this.entries){item.mesh.position.copy(item.position);item.mesh.matrix.copy(item.matrix);}
      this.viewer.model.updateWorldMatrix(true,true);
      if(this.explosion)for(const item of this.entries){const target=item.world.clone().addScaledVector(item.offset,this.explosion/100);item.mesh.parent.worldToLocal(target);item.mesh.position.copy(target);item.mesh.updateMatrix();item.mesh.updateWorldMatrix(false,true);}
    }
    this.refresh();
    if(this.viewer.physicsState)this.viewer.emitPhysicsState();
  }
  snapshot() {
    return {mode:this.viewer.mode,explosion:this.explosion,selected:this.selected,visible:this.entries.filter(entry=>entry.mesh.visible).length,total:this.entries.length,
      entries:this.entries.map(entry=>({id:entry.id,body:entry.body,label:entry.label,path:entry.path,visible:entry.mesh.visible,selected:entry.id===this.selected}))};
  }
  publish() {
    if(this.project)savedProjects.set(this.project,{hidden:[...this.hidden],selected:this.selected});
    this.viewer.container.dispatchEvent(new CustomEvent('quest-parts',{bubbles:true,detail:this.snapshot()}));
  }
  dispose() {
    this.viewer.renderer.domElement.removeEventListener('pointerdown',this.down);this.viewer.renderer.domElement.removeEventListener('pointerup',this.up);
    this.entries=[];
  }
}
