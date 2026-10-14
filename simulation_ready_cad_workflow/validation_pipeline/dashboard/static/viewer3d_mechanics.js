import * as THREE from "three";

const names = {handle_pivot:"手柄轴",upper_toggle_pin:"上肘节销",lower_toggle_pin:"下肘节销",ram_guide:"压杆导轨",grip_attachment:"手柄连接",foot_attachment:"压脚连接",central_pivot:"中心轴",left_suspension:"左盘悬挂",right_suspension:"右盘悬挂",crank_bearing:"曲柄轴承",crank_pin:"曲柄销",wrist_pin:"活塞销",piston_guide:"活塞导向",grip_spindle:"握柄轴"};
const kinds = {revolute:"转动",prismatic:"滑动",fixed:"固定",spherical:"球面"};
const vector = (value) => new THREE.Vector3(...(value || [0,0,0]));
const norm = (value) => Math.hypot(...(value || [0,0,0]));
const number = (value) => Math.abs(value)>0&&Math.abs(value)<.001?Number(value).toExponential(1):Number(value).toLocaleString("zh-CN",{maximumFractionDigits:3});

function dispose(group) {
  group?.traverse((node) => { node.geometry?.dispose(); if(node.material) for(const mat of [node.material].flat()) { mat.map?.dispose(); mat.dispose(); } });
  group?.removeFromParent();
}

function treeSegments(nodes) {
  if(nodes.length<2) return [];
  const used=new Set([0]), result=[];
  while(used.size<nodes.length) {
    let best=null;
    for(const a of used) for(let b=0;b<nodes.length;b++) if(!used.has(b)) {
      const distance=nodes[a].position.distanceToSquared(nodes[b].position);
      if(!best || distance<best.distance) best={a,b,distance};
    }
    if(!best) break;
    used.add(best.b);
    if(best.distance>1e-12) result.push({start:nodes[best.a].position,end:nodes[best.b].position,intensity:Math.min(nodes[best.a].intensity,nodes[best.b].intensity)});
  }
  return result;
}

export function mechanicsForces(frame, branch, metric="force") {
  const evidence=frame?.force_evidence;
  const spots=new Map(), segments=new Map(), pathNodes=new Map(), vectors=[];
  if(!evidence?.available) return {spots,segments,paths:new Map(),vectors,available:false,maxForce:0,maxTorque:0};
  const ceiling=Number(branch?.maxima?.[metric==="force"?"force_n":"torque_nm"]) || 1;
  let maxForce=0,maxTorque=0;
  const add=(body,point,force,torque,id,kind) => {
    if(!body || !point) return;
    const f=norm(force),t=norm(torque),m=metric==="force"?f:t;
    maxForce=Math.max(maxForce,f);maxTorque=Math.max(maxTorque,t);
    const position=vector(point);
    if(f>1e-10 || t>1e-10) {
      const items=pathNodes.get(body)||[];
      if(!items.some(item=>item.position.distanceToSquared(position)<1e-12)) items.push({position,intensity:1,id});
      pathNodes.set(body,items);
      vectors.push({body,position,force:vector(force),torque:vector(torque),kind,id});
    }
    if(m>1e-10) {
      const items=spots.get(body)||[];
      items.push({position,intensity:Math.sqrt(Math.min(1,m/ceiling)),id,kind});
      spots.set(body,items);
    }
  };
  for(const joint of evidence.joint_reactions||[]) if(joint.active!==false) {
    add(joint.body_a,joint.point_a_abs_m,joint.force_a_abs_n,joint.torque_a_abs_nm,`${joint.joint_id}:a`,"joint");
    add(joint.body_b,joint.point_b_abs_m,joint.force_b_abs_n,joint.torque_b_abs_nm,`${joint.joint_id}:b`,"joint");
  }
  for(const [i,contact] of (evidence.contacts||[]).entries()) {
    add(contact.body_a,contact.point_a_abs_m,contact.force_a_abs_n,contact.torque_a_abs_nm,`contact:${i}:a`,"contact");
    add(contact.body_b,contact.point_b_abs_m,contact.force_b_abs_n,contact.torque_b_abs_nm,`contact:${i}:b`,"contact");
  }
  for(const [i,load] of (evidence.external_loads||[]).entries()) add(load.body_id,load.point_abs_m,load.force_abs_n,load.torque_abs_nm,`load:${i}`,load.kind||"load");
  for(const [body,items] of spots) { items.sort((a,b)=>b.intensity-a.intensity);segments.set(body,treeSegments(items)); }
  return {spots,segments,paths:new Map([...pathNodes].map(([body,items])=>[body,treeSegments(items)])),vectors,available:true,maxForce,maxTorque};
}

function labelSprite(text) {
  const canvas=document.createElement("canvas");canvas.width=112;canvas.height=64;
  const ctx=canvas.getContext("2d");ctx.fillStyle="#183f54";ctx.beginPath();ctx.roundRect(4,5,104,52,13);ctx.fill();
  ctx.fillStyle="white";ctx.font="bold 33px sans-serif";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(text,56,32);
  const result=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),depthTest:false,depthWrite:false}));
  result.renderOrder=25;return result;
}

export function setupMechanics(viewer) {
  if(!viewer.replay?.connections?.length) return;
  const group=new THREE.Group();group.name="joint-operation-regions";viewer.scene.add(group);
  const toolbar=document.createElement("div");toolbar.className="studio-mechanics-bar";
  toolbar.innerHTML='<div class="mechanics-switches"><button data-mechanics-regions aria-pressed="true">关节作用域：开</button><button data-mechanics-heat aria-pressed="true">热力图：开</button><button data-mechanics-flow aria-pressed="true">传力线：开</button><select data-mechanics-joint aria-label="选择关节作用域"><option value="">全部关节</option></select><select data-mechanics-metric aria-label="选择热力图量"><option value="force">力 · N</option><option value="torque">力矩 · N·m</option></select></div><div class="mechanics-description"><span class="mechanics-colors"></span><span data-mechanics-status>正在读取受力记录</span><span title="线条连接同一刚体上的真实受力点，用于解释连接关系；不表示材料内部应力轨迹。">连线为刚体传力示意 · 非应力</span></div><div data-mechanics-detail hidden></div>';
  viewer.container.after(toolbar);
  toolbar.querySelector(".mechanics-description > span:last-child").title="直箭头表示力，紫色弧箭头表示力矩。红色为关节A侧反力，橙色为接触力，绿色为已施加载荷；数据均来自本次Chrono记录。虚线连接同一刚体上的受力点，是传力关系示意，不是应力轨迹。";
  const state={regions:true,heat:true,flow:true,metric:"force",joint:"",group,toolbar,parts:[],labels:[],drawings:null};
  viewer.mechanics=state;
  const select=toolbar.querySelector("[data-mechanics-joint]");
  for(const [index,connection] of viewer.replay.connections.entries()) {
    const option=document.createElement("option");option.value=connection.id;option.textContent=`J${index+1} ${names[connection.id]||connection.id.replaceAll("_"," ")} · ${kinds[connection.type]||connection.type}`;select.append(option);
    for(const side of ["a","b"]) {
      const region=connection[`region_${side}`];if(!region?.points) continue;
      const a=vector(region.points[0]),b=vector(region.points[1]),delta=b.clone().sub(a),length=delta.length();
      const geometry=length>1e-12?new THREE.CapsuleGeometry(region.radius,length,5,18):new THREE.SphereGeometry(region.radius,18,12);
      const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:side==="a"?0x179fbb:0xf3ad35,transparent:true,opacity:.18,depthWrite:false,side:THREE.DoubleSide}));
      mesh.name=`region-${connection.id}-${side}`;
      const wire=new THREE.LineSegments(new THREE.WireframeGeometry(geometry),new THREE.LineBasicMaterial({color:side==="a"?0x118eaa:0xd58e15,transparent:true,opacity:.55,depthWrite:false}));mesh.add(wire);
      const local=new THREE.Group();local.position.copy(a.clone().add(b).multiplyScalar(.5));
      if(length>1e-12)local.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());local.add(mesh);
      const bodyGroup=new THREE.Group();bodyGroup.add(local);group.add(bodyGroup);
      state.parts.push({connection,side,body:connection[`body_${side}`],group:bodyGroup,local,mesh,wire});
    }
    const label=labelSprite(`J${index+1}`);group.add(label);
    const leader=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]),new THREE.LineBasicMaterial({color:0x567b89,transparent:true,opacity:.75,depthTest:false}));leader.renderOrder=20;group.add(leader);
    state.labels.push({connection,label,leader,anchor:new THREE.Vector3()});
  }
  for(const key of ["regions","heat","flow"]) viewer.listenPhysics(toolbar.querySelector(`[data-mechanics-${key}]`),"click",()=>{state[key]=!state[key];viewer.applyPhysicsFrame();});
  viewer.listenPhysics(select,"change",()=>{state.joint=select.value;viewer.applyPhysicsFrame();});
  viewer.listenPhysics(toolbar.querySelector("[data-mechanics-metric]"),"change",event=>{state.metric=event.target.value;viewer.applyPhysicsFrame();});
  viewer.container.dataset.connectionCount=String(viewer.replay.connections.length);
  viewer.container.dataset.regionCount=String(state.parts.length);
  requestAnimationFrame(()=>{if(!viewer.disposed){viewer.resize();viewer.fitView();viewer.applyPhysicsFrame();}});
}

export function updateMechanics(viewer,frame,branch,forces) {
  const state=viewer.mechanics;if(!state)return;
  const selected=state.joint;
  for(const part of state.parts) {
    const pose=frame.body_poses?.[part.body],q=pose?.rotation_abs_quat_wxyz||[1,0,0,0];
    part.group.position.copy(vector(pose?.position_abs_m));part.group.quaternion.set(q[1],q[2],q[3],q[0]);
    part.group.visible=state.regions&&viewer.bodyIsVisible(part.body)&&(!selected||selected===part.connection.id);
    const active=frame.joint_active?.[part.connection.id]!==false;
    part.mesh.material.opacity=active ? 0.18 : 0.06;part.wire.material.color.set(active?(part.side==="a"?0x118eaa:0xd58e15):0xbd3d35);
  }
  state.group.updateMatrixWorld(true);
  for(const item of state.labels) {
    const {connection,label,leader}=item;
    const parts=state.parts.filter(part=>part.connection.id===connection.id&&part.group.visible);
    label.visible=parts.length>0;leader.visible=label.visible;if(!parts.length)continue;
    const center=new THREE.Vector3();for(const part of parts)center.add(part.local.getWorldPosition(new THREE.Vector3()));center.multiplyScalar(1/parts.length);
    item.anchor.copy(center);
  }
  dispose(state.drawings);state.drawings=new THREE.Group();state.drawings.name="recorded-force-vectors-and-body-paths";viewer.scene.add(state.drawings);
  if(state.flow&&forces.available) {
    for(const [body,paths] of forces.paths) if(viewer.bodyIsVisible(body)) for(const path of paths) {
      const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([path.start,path.end]),new THREE.LineDashedMaterial({color:0x96337f,dashSize:viewer.axisLength*.025,gapSize:viewer.axisLength*.013,transparent:true,opacity:.7,depthTest:false}));
      line.computeLineDistances();line.renderOrder=11;state.drawings.add(line);
    }
    // Every vector comes from a recorded wrench. Opposite body reactions are
    // retained in the data; draw one side per joint to keep the scene legible.
    for(const item of forces.vectors) if(viewer.bodyIsVisible(item.body)&&!item.id.endsWith(":b")) {
      const magnitude=item.force.length();
      if(magnitude>1e-8) {
        const ratio=Math.sqrt(Math.min(1,magnitude/Math.max(branch.maxima?.force_n||magnitude,1e-12)));
        const length=viewer.axisLength*(.06+.25*ratio);
        const arrow=new THREE.ArrowHelper(item.force.clone().normalize(),item.position,length,item.kind==="joint"?0xc44536:item.kind==="contact"?0xe58b1d:0x168660,length*.23,length*.1);
        arrow.line.material.depthTest=false;arrow.cone.material.depthTest=false;arrow.renderOrder=14;state.drawings.add(arrow);
      }
      if(item.torque.length()>1e-8) {
        const rotation=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1),item.torque.clone().normalize());
        const radius=viewer.axisLength*.08,points=Array.from({length:27},(_,i)=>new THREE.Vector3(radius*Math.cos(i/26*5),radius*Math.sin(i/26*5),0).applyQuaternion(rotation).add(item.position));
        const arc=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0x7441ab,depthTest:false}));arc.renderOrder=14;state.drawings.add(arc);
        const direction=new THREE.Vector3(-Math.sin(5),Math.cos(5),0).applyQuaternion(rotation);
        const head=new THREE.ArrowHelper(direction,points.at(-1),radius*.4,0x7441ab,radius*.35,radius*.16);head.line.material.depthTest=false;head.cone.material.depthTest=false;state.drawings.add(head);
      }
    }
  }
  const recorded=Boolean(branch.field_availability?.joint_reactions);
  for(const key of ["regions","heat","flow"]) {const button=state.toolbar.querySelector(`[data-mechanics-${key}]`);button.disabled=key!=="regions"&&!recorded;button.setAttribute("aria-pressed",String(state[key]&&!button.disabled));button.textContent=`${{regions:"关节作用域",heat:"热力图",flow:"传力线"}[key]}：${button.disabled?"待数据":state[key]?"开":"关"}`;}
  state.toolbar.querySelector("[data-mechanics-status]").textContent=forces.available?`低 → 高 · 相对色阶　当前最大力 ${number(forces.maxForce)} N / 力矩 ${number(forces.maxTorque)} N·m`:(!recorded?"此记录尚未保存反力；当前可查看关节作用域":frame.time_s===0?"初始姿态尚无反力；播放后查看":"当前帧未记录反力");
  const detail=state.toolbar.querySelector("[data-mechanics-detail]");detail.hidden=!selected;
  if(selected) {
    const connection=viewer.replay.connections.find(item=>item.id===selected),reaction=frame.force_evidence?.joint_reactions?.find(item=>item.joint_id===selected);
    detail.textContent=`${names[selected]||selected}：${kinds[connection.type]||connection.type}理想约束；两侧区域半径 ${number(connection.region_a.radius*1000)} / ${number(connection.region_b.radius*1000)} mm。${reaction?`反力 ${number(norm(reaction.force_a_abs_n))} N，反力矩 ${number(norm(reaction.torque_a_abs_nm))} N·m。`:""}作用域只控制连接是否有效，不代表弹性材料。`;
  }
  viewer.container.dataset.forceFrameAvailable=String(forces.available);
  viewer.container.dataset.forceVectorCount=String(state.flow?forces.vectors.length:0);
  viewer.container.dataset.forcePathCount=String(state.flow?[...forces.paths.values()].reduce((sum,items)=>sum+items.length,0):0);
}

export function layoutMechanicsLabels(viewer) {
  if(!viewer.mechanics)return;
  const width=viewer.container.clientWidth,height=viewer.container.clientHeight,placed=[];
  if(width<1||height<1)return;
  viewer.camera.updateMatrixWorld();
  for(const {label,leader,anchor} of viewer.mechanics.labels) {
    if(!label.visible)continue;
    const projected=anchor.clone().project(viewer.camera),depth=-anchor.clone().applyMatrix4(viewer.camera.matrixWorldInverse).z;
    if(depth<=0){label.visible=false;leader.visible=false;continue;}
    const x=(projected.x+1)*width/2,y=(1-projected.y)*height/2;
    let chosen={x,y:y-20};
    for(const [dx,dy] of [[0,-20],[-34,-20],[34,-20],[-48,4],[48,4],[0,-48],[0,30],[-45,30],[45,30]]) {
      const point={x:Math.max(23,Math.min(width-23,x+dx)),y:Math.max(15,Math.min(height-15,y+dy))};
      if(!placed.some(other=>Math.abs(other.x-point.x)<44&&Math.abs(other.y-point.y)<26)){chosen=point;break;}
    }
    placed.push(chosen);
    label.position.set(chosen.x/width*2-1,1-chosen.y/height*2,projected.z).unproject(viewer.camera);
    const unitsPerPixel=2*Math.tan(THREE.MathUtils.degToRad(viewer.camera.fov)/2)*depth/height;
    label.scale.set(38.5*unitsPerPixel,22*unitsPerPixel,1);
    const vertices=leader.geometry.attributes.position;
    vertices.setXYZ(0,anchor.x,anchor.y,anchor.z);vertices.setXYZ(1,label.position.x,label.position.y,label.position.z);vertices.needsUpdate=true;
    leader.geometry.computeBoundingSphere();
  }
}
