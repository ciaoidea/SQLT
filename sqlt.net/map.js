'use strict';
if (location.protocol === 'file:') {
  location.replace('http://127.0.0.1:8090/');
} else {
const $ = id => document.getElementById(id);
const apiBase='/';
$('close-details').addEventListener('click',()=>document.body.classList.remove('inspecting'));
let currentCatalog=null,currentDataset='memory',browseSpec={dataset:'memory',filters:[],offset:0},lastData=null,viewMode='table',detailRequest=0;
let graphSource='empty';
let cy, selected = null, request = 0, space = null, rotating = true, pendingClarification = null;
// Hierarchical hubs accumulate all distinct descendants, not just immediate links.
function aggregateCounts(nodes, links) {
 const children=new Map(nodes.map(n=>[n.id,[]])),degree=new Map(nodes.map(n=>[n.id,0]));
 for(const link of links){children.get(link.source)?.push(link.target);degree.set(link.source,(degree.get(link.source)||0)+1);degree.set(link.target,(degree.get(link.target)||0)+1);}
 const hierarchical=nodes.some(n=>n.kind==='group');
 return new Map(nodes.map(node=>{
  if(!hierarchical)return [node.id,Math.max(1,degree.get(node.id)||0)];
  const seen=new Set([node.id]),stack=[...(children.get(node.id)||[])];
  while(stack.length){const id=stack.pop();if(seen.has(id))continue;seen.add(id);stack.push(...(children.get(id)||[]));}
  return [node.id,Math.max(1,seen.size-1)];
 }));
}
// Disk area represents the accumulated node count; leaf nodes retain a visible minimum.
function nodeRadius(weight) { return 3.5 * Math.sqrt(Math.max(1, weight || 0)); }
// Visual connectivity scale, not a physical stellar temperature model.
function stellarStyle(weight, maximum) {
 const density=Math.log1p(Math.max(0,(weight||1)-1))/Math.log1p(Math.max(1,(maximum||1)-1));
 const stops=[[255,76,48],[255,200,85],[91,161,255],[255,255,255]];
 const t=Math.max(0,Math.min(1,density))*3,i=Math.min(2,Math.floor(t)),f=t-i;
 const rgb=stops[i].map((c,k)=>Math.round(c+(stops[i+1][k]-c)*f));
 return {rgb,color:`rgb(${rgb.join(',')})`,brightness:.38+.62*density};
}
function text(tag, value, parent) { const el = document.createElement(tag); el.textContent = value; parent.appendChild(el); return el; }
function layout() { sync3D(); }
// Build a radial spanning forest for layout only; all original links remain drawn.
function radialPositions(nodes, links, weights) {
 const adjacent=new Map(nodes.map(n=>[n.id,[]]));
 for(const edge of links){adjacent.get(edge.source)?.push(edge.target);adjacent.get(edge.target)?.push(edge.source);}
 const priority=[...nodes].sort((a,b)=>(a.group_level ?? 99)-(b.group_level ?? 99)||(weights.get(b.id)||0)-(weights.get(a.id)||0));
 const children=new Map(nodes.map(n=>[n.id,[]])),seen=new Set(),roots=[];
 for(const node of priority){
  if(seen.has(node.id))continue;roots.push(node.id);seen.add(node.id);const queue=[node.id];
  while(queue.length){const id=queue.shift();for(const next of adjacent.get(id)||[])if(!seen.has(next)){seen.add(next);children.get(id).push(next);queue.push(next);}}
 }
 // Fibonacci directions cover a sphere rather than a planar ring.
 // Rotate each local sphere deterministically so nested branches do not align.
 function directions(count,key){
  let hash=2166136261;for(const c of key)hash=Math.imul(hash^c.charCodeAt(0),16777619)>>>0;
  const az=hash/4294967296*Math.PI*2,tilt=.45+(hash%997)/997;
  return Array.from({length:count},(_,i)=>{
   const z=1-2*(i+.5)/count,r=Math.sqrt(Math.max(0,1-z*z)),a=i*Math.PI*(3-Math.sqrt(5));
   const x=r*Math.cos(a),y=r*Math.sin(a),yy=y*Math.cos(tilt)-z*Math.sin(tilt),zz=y*Math.sin(tilt)+z*Math.cos(tilt);
   return {x:x*Math.cos(az)-yy*Math.sin(az),y:x*Math.sin(az)+yy*Math.cos(az),z:zz};
  });
 }
 const extent=new Map(),orbits=new Map(),rays=new Map();
 function measure(id){
  const list=children.get(id),soma=nodeRadius(weights.get(id))+22;
  if(!list.length){extent.set(id,soma);return soma;}
  const sizes=list.map(child=>measure(child)),vectors=directions(list.length,id);
  // Bounding spheres keep sibling subtrees apart in all three dimensions.
  let orbit=soma+Math.max(...sizes)+45;
  for(let i=0;i<list.length;i++)for(let j=i+1;j<list.length;j++){
   const a=vectors[i],b=vectors[j],distance=Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z);
   orbit=Math.max(orbit,(sizes[i]+sizes[j]+24)/Math.max(distance,.001));
  }
  orbits.set(id,orbit);rays.set(id,vectors);
  const radius=orbit+Math.max(...sizes);extent.set(id,radius);return radius;
 }
 roots.forEach(measure);
 const positions=new Map();
 function place(id,x,y,z,cluster){
  positions.set(id,{x,y,z,cluster});
  children.get(id).forEach((child,i)=>{
   const v=rays.get(id)[i],r=orbits.get(id);
   place(child,x+v.x*r,y+v.y*r,z+v.z*r,cluster);
  });
 }
 const rootRays=directions(roots.length,'forest');
 let rootOrbit=0;
 for(let i=0;i<roots.length;i++)for(let j=i+1;j<roots.length;j++){
  const a=rootRays[i],b=rootRays[j];
  rootOrbit=Math.max(rootOrbit,(extent.get(roots[i])+extent.get(roots[j])+90)/Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z));
 }
 roots.forEach((root,i)=>{const v=rootRays[i];place(root,v.x*rootOrbit,v.y*rootOrbit,v.z*rootOrbit,i);});
 if(positions.size){
  const all=[...positions.values()],center={};
  for(const axis of ['x','y','z'])center[axis]=(Math.min(...all.map(p=>p[axis]))+Math.max(...all.map(p=>p[axis])))/2;
  let radius=1;for(const p of all)radius=Math.max(radius,Math.hypot(p.x-center.x,p.y-center.y,p.z-center.z));
  const scale=355/radius;for(const p of all)for(const axis of ['x','y','z'])p[axis]=(p[axis]-center[axis])*scale;
 }
 return positions;
}
function constellation(nodes,links){
 const weights=aggregateCounts(nodes,links),positions=radialPositions(nodes,links,weights);
 const degree=new Map(nodes.map(n=>[n.id,0]));for(const edge of links){degree.set(edge.source,(degree.get(edge.source)||0)+1);degree.set(edge.target,(degree.get(edge.target)||0)+1);}
 return nodes.map(n=>{const aggregate_count=weights.get(n.id);cy.getElementById(n.id).data('aggregate_count',aggregate_count);return {...n,...positions.get(n.id),degree:degree.get(n.id),aggregate_count};});
}
function caption(n){
 if(n.kind==='group')return `${n.group_role}: ${n.label}`;
 if(n.kind==='event')return n.label==='branch connectivity'?'Connectivity':n.label;
 const fields=n.label.split(' | ');return fields.length>1?fields[1]:n.label;
}
function inspect(node) {
  if(node.data('record')){showObject(node.data('dataset'),node.data('record'));return;}
  document.body.classList.add('inspecting');
  selected = node; const d = node.data(), box = $('details'); box.replaceChildren();
  $('kind').textContent = d.kind === 'event' ? 'Fact' : d.kind === 'query' ? 'Query' : 'Entity';
  text('h1',d.label,box); $('expand').hidden = d.kind !== 'entity';
  if (d.event) {
    const e=d.event; if (e.negated) text('p','This fact is recorded as negated.',box);
    if(e.time) text('p',e.time,box);
    const dl=document.createElement('dl');box.appendChild(dl);
    for(const b of e.roles){text('dt',b.role,dl);text('dd',b.value,dl);}
    text('h2','Original source',box);text('pre',e.source || 'Source not included in the live trace.',box);
    if(!e.source&&e.event_id){const button=text('button','Read source #'+e.event_id,box);button.onclick=()=>showObject('memory',e);}
  } else if (d.kind === 'group') {text('p',`${d.members_count} distinct entities · ${d.aggregate_count || 0} total descendant nodes. Grouped by ${d.group_role}: ${d.group_value}. Grouping uses stored fields.`,box);
  } else if (d.kind === 'query') {text('p','Dashed links indicate query similarity, not proven relationships.',box);}
  else {text('p',`${node.connectedEdges('[kind="binding"]').length} connections in this view. Expand to find related facts.`,box);}
  cy.elements().removeClass('focus');node.closedNeighborhood().addClass('focus');
}
async function graphRequest(url,token,merge=false){
 graphSource='query';
 const dynamic=!!window.CortexLive?.active();
 if(!dynamic){const response=await fetch(url,{cache:'no-store'});return {response,data:await response.json()};}
 const path=url+(url.includes('?')?'&':'?')+'live=1';
 const response=await fetch(path,{cache:'no-store'});
 if(!response.ok || !response.headers.get('content-type')?.includes('application/x-ndjson'))return {response,data:await response.json()};
 const previous=cy.json().elements;
 let cleared=merge;
 window.CortexLive.beginQuery();
 const reader=response.body.getReader(),decoder=new TextDecoder();
 let buffer='',final=null,layoutTimer=null,dropped=0;
 const apply=packet=>{
  if(token!==request)return;
  if(packet.type==='result'){final=packet.data;dropped=packet.dropped_progress||0;return;}
  if(packet.type==='error')throw Error(packet.error);
  if(packet.type!=='progress'||!window.CortexLive.active())return;
  const elements=packet.elements||[];
  if(elements.length&&!cleared){cy.elements().remove();cleared=true;}
  cy.batch(()=>{
   for(const element of elements){const d=element.data;if(d.source)continue;const old=cy.getElementById(d.id);if(old.length)old.data(d);else if(cy.nodes().length<380)cy.add(element);}
   for(const element of elements){const d=element.data;if(!d.source)continue;if(cy.getElementById(d.source).length&&cy.getElementById(d.target).length&&!cy.getElementById(d.id).length)cy.add(element);}
  });
  window.CortexLive.pipeline(packet);
  if(elements.length && !layoutTimer)layoutTimer=setTimeout(()=>{layoutTimer=null;if(token===request)layout();},75);
 };
 try{
  while(true){const {done,value}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
   let split;while((split=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,split);buffer=buffer.slice(split+1);if(line)apply(JSON.parse(line));}
   if(done)break;
  }
  if(!final)throw Error('Query stream ended before its final result');
  return {response,data:final};
 }catch(error){if(token===request){cy.elements().remove();cy.add(previous);layout();}throw error;}
 finally{clearTimeout(layoutTimer);if(token===request)window.CortexLive.endQuery(dropped);reader.releaseLock();}
}

async function load(mode='initial',q='',merge=false){
 $('clarification').hidden=true;
 const token=++request; $('status').textContent='Working…'; $('loading').hidden=false; $('expand').disabled=true;$('submit-query').disabled=true;
 $('loading').textContent=mode==='natural'?'Planning query…':'Reading memory…';
 $('sql-query').textContent='Working…';$('query-meta').textContent='';
 try{
  const {response,data}=await graphRequest(apiBase+'api/graph?'+new URLSearchParams({mode,q}),token,merge);if(!response.ok)throw Error(data.error || 'Server error');if(token!==request)return;
  if(data.plan?.mode==='clarify'){
   showQuery(data);
   $('status').textContent='Clarification needed';
   pendingClarification={question:q,clarification:data.plan.clarification || data.note || 'Please clarify your query.'};
   $('clarification-message').textContent=pendingClarification.clarification;
   $('query').value='';
   $('query').placeholder='Reply to the clarification…';
   $('clarification').hidden=false;
   $('loading').hidden=true;
   $('query').focus();
   return;
  }
  pendingClarification=null;
  $('query').placeholder='Show, filter, group, connect…';
  if(!merge){document.body.classList.remove('inspecting');cy.elements().remove();$('details').replaceChildren();text('h1',q || 'Memory', $('details'));text('p','Select a node to inspect its source.', $('details'));$('expand').hidden=true;selected=null;}
  const incomingNodes=data.elements.filter(e=>!e.data.source && !cy.getElementById(e.data.id).length);
  const capacity=Math.max(0,380-cy.nodes().length);const accepted=incomingNodes.slice(0,capacity);
  cy.batch(()=>{cy.add(accepted);for(const el of data.elements){if(el.data.source && !cy.getElementById(el.data.id).length && cy.getElementById(el.data.source).length && cy.getElementById(el.data.target).length)cy.add(el);}});
  layout();showQuery(data);renderRecords(data);
  const levels=data.group_levels || data.plan?.group_by || [];
  $('grouping-path').hidden=!levels.length;
  $('grouping-path').textContent='Visual hubs: '+levels.map(role=>role[0].toUpperCase()+role.slice(1)).concat('Branch').join(' → ');
  const limited=data.partial || incomingNodes.length>accepted.length;
  $('status').textContent=data.dataset && data.dataset!=='memory'?`${data.records?.length || 0} rows · ${data.total ?? 0} matching ${data.dataset}${limited?' · partial view':''}`:`${cy.nodes().length} nodes · ${cy.edges().length} connections${limited?' · partial view':''}${data.total!==undefined?' · '+data.total+' active facts':''}`;
  $('status').title=data.note || '';
  if(data.title){$('details').replaceChildren();text('h1',data.title,$('details'));text('p',`${data.matched_events ?? data.events} record${data.entities!==undefined?' · '+data.entities+' entities · '+data.groups+' groups':''}. ${data.note || ''}`,$('details'));}
  if(!data.events){text('p','No results. Try another query.', $('details'));}
 }catch(error){if(token===request){$('status').textContent=error.message;$('sql-query').textContent='Query failed.';}}
 finally{if(token===request){if($('status').textContent!=='Clarification needed')$('loading').hidden=true;$('expand').disabled=false;$('submit-query').disabled=false;}}
}


function canvasSpace(container){
 const canvas=document.createElement('canvas');container.appendChild(canvas);
 const ctx=canvas.getContext('2d');
 let w=container.clientWidth,h=container.clientHeight,data={nodes:[],links:[]},angleX=.22,angleY=.15,angleZ=0,zoom=1,panX=0,panY=0,points=[],drag=null,moved=0,dirty=true,running=false,last=0,raf,hovered=null;
 const controls={autoRotate:true,autoRotateSpeed:.6};
 function draw(){
  ctx.clearRect(0,0,w,h);
  const backdrop=ctx.createRadialGradient(w*.47,h*.45,20,w*.47,h*.45,Math.max(w,h)*.7);backdrop.addColorStop(0,'#050811');backdrop.addColorStop(.55,'#02040a');backdrop.addColorStop(1,'#000000');ctx.fillStyle=backdrop;ctx.fillRect(0,0,w,h);
  const ax=Math.cos(angleX),bx=Math.sin(angleX),ay=Math.cos(angleY),by=Math.sin(angleY);
  const scale=Math.min(w/850,h/800)*zoom;
  const horizontalSpread=Math.max(1,Math.min(2.3,w/Math.max(h,1)*.82));
  const az=Math.cos(angleZ),bz=Math.sin(angleZ);
  points=data.nodes.map(n=>{
   const nx=n.x*az-n.y*bz,ny=n.x*bz+n.y*az;
   const x=nx*ay+n.z*by,z=-nx*by+n.z*ay;
   const y=ny*ax-z*bx,depth=ny*bx+z*ax;
   const perspective=850/Math.max(250,850+depth);
   return {n,x:w/2+panX+x*scale*horizontalSpread*perspective,y:h/2+panY+y*scale*perspective,z:depth,r:nodeRadius(n.aggregate_count)*perspective,opacity:Math.max(.3,Math.min(1,1-depth/900))};
  }).sort((a,b)=>b.z-a.z);
  const byId=new Map(points.map(p=>[p.n.id,p]));
  const maximum=Math.max(1,...data.nodes.map(n=>n.aggregate_count||1));
  // Soft cluster envelopes are drawn behind links and reprojected with the nodes.
  // Use terminal groups to avoid a single company cloud swallowing every region.
  const childIds=new Map(data.nodes.map(n=>[n.id,[]]));
  for(const link of data.links)childIds.get(link.source)?.push(link.target);
  const groups=data.nodes.filter(n=>n.kind==='group' && !(childIds.get(n.id)||[]).some(id=>byId.get(id)?.n.kind==='group'));
  const clouds=groups.map(group=>[group.id,...(childIds.get(group.id)||[])]);
  if(!groups.length){
   const clusters=new Map();
   for(const point of points){const key=point.n.cluster;if(!clusters.has(key))clusters.set(key,[]);clusters.get(key).push(point.n.id);}
   clouds.push(...clusters.values());
  }
  for(const ids of clouds){
   const members=ids.map(id=>byId.get(id)).filter(Boolean);if(members.length<2)continue;
   const padding=24;
   const samples=members.flatMap(p=>{const r=p.r+padding;return Array.from({length:8},(_,i)=>{const a=i*Math.PI/4;return {x:p.x+Math.cos(a)*r,y:p.y+Math.sin(a)*r};});}).sort((a,b)=>a.x-b.x||a.y-b.y);
   const cross=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
   const lower=[],upper=[];
   for(const p of samples){while(lower.length>1&&cross(lower[lower.length-2],lower[lower.length-1],p)<=0)lower.pop();lower.push(p);}
   for(const p of [...samples].reverse()){while(upper.length>1&&cross(upper[upper.length-2],upper[upper.length-1],p)<=0)upper.pop();upper.push(p);}
   const hull=[...lower.slice(0,-1),...upper.slice(0,-1)];if(hull.length<3)continue;
   ctx.save();
   const active=!hovered||ids.includes(hovered);
   ctx.globalAlpha=active?1:.35;
   const cloud=stellarStyle(Math.max(...members.map(p=>p.n.aggregate_count||1)),maximum),rgb=cloud.rgb.join(',');
   ctx.fillStyle=`rgba(${rgb},0.065)`;ctx.strokeStyle=`rgba(${rgb},0.16)`;ctx.lineWidth=1;
   ctx.shadowColor=`rgba(${rgb},0.3)`;ctx.shadowBlur=28;
   const last=hull[hull.length-1],first=hull[0];ctx.beginPath();ctx.moveTo((last.x+first.x)/2,(last.y+first.y)/2);
   hull.forEach((p,i)=>{const next=hull[(i+1)%hull.length];ctx.quadraticCurveTo(p.x,p.y,(p.x+next.x)/2,(p.y+next.y)/2);});
   ctx.closePath();ctx.fill();ctx.shadowBlur=0;ctx.stroke();ctx.restore();
  }
  const nearby=new Set(hovered?[hovered,...data.links.filter(l=>l.source===hovered||l.target===hovered).flatMap(l=>[l.source,l.target])]:[]);
  const hierarchical=data.nodes.some(n=>n.kind==='group');
  for(const link of data.links){
   const a=byId.get(typeof link.source==='object'?link.source.id:link.source),b=byId.get(typeof link.target==='object'?link.target.id:link.target);if(!a||!b)continue;
   const active=!hovered||a.n.id===hovered||b.n.id===hovered;
   ctx.globalAlpha=active?.92:.25;
   const gradient=ctx.createLinearGradient(a.x,a.y,b.x,b.y);gradient.addColorStop(0,stellarStyle(a.n.aggregate_count,maximum).color);gradient.addColorStop(1,stellarStyle(b.n.aggregate_count,maximum).color);ctx.strokeStyle=gradient;
   ctx.lineWidth=hovered&&active?2:1.15;ctx.setLineDash(link.kind==='similarity'?[4,5]:[]);
   const dx=b.x-a.x,dy=b.y-a.y,bend=hierarchical?.035:.12;
   const path=()=>{ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo((a.x+b.x)/2-dy*bend,(a.y+b.y)/2+dx*bend,b.x,b.y);};
   const opacity=ctx.globalAlpha,weight=ctx.lineWidth;
   if(link.kind!=='similarity'){ctx.globalAlpha=opacity*.08;ctx.lineWidth=14;path();ctx.stroke();ctx.globalAlpha=opacity*.14;ctx.lineWidth=5;path();ctx.stroke();}
   ctx.globalAlpha=opacity;ctx.lineWidth=weight;path();ctx.stroke();
  }
  ctx.setLineDash([]);
  for(const p of points){
   const star=stellarStyle(p.n.aggregate_count,maximum),color=star.color,rgb=star.rgb.join(',');
   ctx.globalAlpha=(hovered&&!nearby.has(p.n.id)?.2:1)*p.opacity*star.brightness;
   const halo=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,p.r*3.8);
   halo.addColorStop(0,`rgba(${rgb},0.8)`);halo.addColorStop(.22,`rgba(${rgb},0.32)`);halo.addColorStop(1,`rgba(${rgb},0)`);
   ctx.fillStyle=halo;ctx.beginPath();ctx.arc(p.x,p.y,p.r*3.8,0,Math.PI*2);ctx.fill();
   const core=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,p.r);
   core.addColorStop(0,star.brightness>.8?'#ffffff':color);core.addColorStop(.4,color);core.addColorStop(1,`rgba(${rgb},0.08)`);
   ctx.fillStyle=core;ctx.strokeStyle=color;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fill();
   if(p.n.id===hovered){ctx.globalAlpha=1;ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.x,p.y,p.r+5,0,Math.PI*2);ctx.stroke();}
  }
  const occupied=[];
  for(const p of [...points].sort((a,b)=>(b.n.aggregate_count||0)-(a.n.aggregate_count||0))){
   ctx.globalAlpha=hovered&&!nearby.has(p.n.id)?.3:Math.max(.6,p.opacity);
   const fontSize=Math.min(13,9+Math.log2(Math.max(1,p.n.aggregate_count||1))*.6);ctx.font=(p.n.aggregate_count>5?'600 ':'')+fontSize+'px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';
   const full=caption(p.n),label=full.length>32?full.slice(0,30)+'…':full;
   const tw=ctx.measureText(label).width+10,th=16;
   const offsets=[[0,p.r+15],[0,-p.r-15],[tw/2+p.r+7,0],[-tw/2-p.r-7,0],[0,p.r+29],[0,-p.r-29]];
   let chosen;
   for(const [dx,dy] of offsets){const r={x:p.x+dx-tw/2,y:p.y+dy-th/2,w:tw,h:th};if(!occupied.some(b=>r.x<b.x+b.w&&r.x+r.w>b.x&&r.y<b.y+b.h&&r.y+r.h>b.y)){chosen=r;break;}}
   chosen=chosen||{x:p.x-tw/2,y:p.y+p.r+7,w:tw,h:th};occupied.push(chosen);p.labelRect=chosen;
   const x=chosen.x+tw/2,y=chosen.y+th/2;
   ctx.strokeStyle='#02040b';ctx.lineWidth=4;ctx.lineJoin='round';ctx.strokeText(label,x,y);ctx.fillStyle=p.n.id===hovered?'#ffffff':p.n.kind==='event'?'#a8b4c8':'#d2dbea';ctx.fillText(label,x,y);
  }
  ctx.globalAlpha=1;window.CortexLive?.render(ctx,points,w,h);dirty=!!window.CortexLive?.animating();
 }
 function frame(t){if(!running)return;if(t-last>32){last=t;if(controls.autoRotate && !hovered && !drag){angleZ+=.006*controls.autoRotateSpeed;dirty=true;}if(dirty || window.CortexLive?.animating())draw();}raf=requestAnimationFrame(frame);}
 function size(){canvas.width=w;canvas.height=h;canvas.style.width=w+'px';canvas.style.height=h+'px';dirty=true;}
 canvas.addEventListener('contextmenu',e=>e.preventDefault());
 canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,button:e.button};moved=0;canvas.setPointerCapture(e.pointerId);});
 function hitAt(e){const rect=canvas.getBoundingClientRect();const x=e.clientX-rect.left,y=e.clientY-rect.top;return [...points].reverse().find(p=>Math.hypot(x-p.x,y-p.y)<p.r+8 || (p.labelRect && x>=p.labelRect.x && x<=p.labelRect.x+p.labelRect.w && y>=p.labelRect.y && y<=p.labelRect.y+p.labelRect.h));}
 canvas.addEventListener('pointerleave',()=>{hovered=null;dirty=true;$('hover-card').hidden=true;});
 canvas.addEventListener('pointermove',e=>{
  if(!drag){const hit=hitAt(e);hovered=hit?hit.n.id:null;dirty=true;canvas.style.cursor=hit?'pointer':'grab';
   const card=$('hover-card');card.hidden=!hit;
   if(hit){$('hover-title').textContent=hit.n.label;$('hover-description').textContent=description(hit.n);
    const rect=container.getBoundingClientRect();card.style.left=Math.max(8,Math.min(e.clientX-rect.left+18,w-340))+'px';card.style.top=Math.max(8,Math.min(e.clientY-rect.top+18,h-card.offsetHeight-12))+'px';}
   return;
  }
  hovered=null;$('hover-card').hidden=true;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;moved+=Math.abs(dx)+Math.abs(dy);if(drag.button===2){panX+=dx;panY+=dy;}else{angleY+=dx*.007;angleX+=dy*.007;}drag.x=e.clientX;drag.y=e.clientY;dirty=true;
 });
 canvas.addEventListener('pointerup',e=>{if(drag&&moved<5){const hit=hitAt(e);if(hit)inspect(cy.getElementById(hit.n.id));}drag=null;});
 canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.15,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));dirty=true;},{passive:false});
 const api={
  graphData(value){if(!value)return data;hovered=null;$('hover-card').hidden=true;data={nodes:constellation(value.nodes,value.links),links:value.links};dirty=true;return api;},
  width(value){w=value;size();return api;},height(value){h=value;size();return api;},
  controls(){return controls;},renderer(){return {setPixelRatio(){}};},
  zoomToFit(){zoom=1;panX=panY=0;dirty=true;return api;},onEngineStop(){return api;},
  pauseAnimation(){running=false;cancelAnimationFrame(raf);return api;},
  resumeAnimation(){if(!running){running=true;raf=requestAnimationFrame(frame);}return api;}
 };
 size();return api;
}

function showQuery(data){
 const traces=data.sql_trace || [];
 const sections=traces.map((t,i)=>{
  const params=(t.parameters || []).map(p=>p && p.type==='VECTOR(384)'?{type:p.type,encoding:p.encoding,sha256:p.sha256}:p);
  return `-- SELECT ${i+1} · current memory namespace\n${t.sql}\n-- Parameters: ${JSON.stringify(params)}`;
 });
 if(data.requested_query)sections.unshift('-- Natural Language Query: '+data.requested_query.replace(/\s+/g,' '));
 if(data.requested_group_levels && JSON.stringify(data.requested_group_levels)!==JSON.stringify(data.group_levels))sections.push('-- Requested grouping: '+data.requested_group_levels.join(' → ')+'\n-- Visual hub layout: '+data.group_levels.join(' → ')+' (one hub per company)');
 if(data.plan){sections.push('-- Validated query plan\n'+JSON.stringify(data.plan,null,2));if(data.plan.group_by && data.plan.group_by.length)sections.push('-- View grouping applied to retrieved records: '+(Array.isArray(data.plan.group_by)?data.plan.group_by.join(' → '):data.plan.group_by)+'; labels derived from stored fields.');}
 $('sql-query').textContent=sections.join('\n\n') || data.note || 'No query executed.';
 const sqlSeconds=traces.reduce((n,t)=>n+(t.seconds || 0),0);
 $('query-meta').textContent=`${traces.length} SELECT · SQL ${sqlSeconds.toFixed(3)}s${data.planning_seconds!==undefined?' · LLM '+data.planning_seconds.toFixed(1)+'s':''}`;
}

function description(d){
 if(d.record)return Object.entries(d.record).filter(([k,v])=>typeof v!=='object').map(([k,v])=>k+': '+v).join(' · ').slice(0,350);
 if(d.kind==='group')return `${d.members_count} distinct entities · ${d.aggregate_count || 0} total descendant nodes · ${d.group_role}: ${d.group_value}`;
 if(d.kind==='query')return 'Semantic search: links indicate similarity, not proof.';
 if(d.event){
  const e=d.event;const summary=e.roles.find(b=>['summary','context','activities','activities_observed','description','topic'].includes(b.role));
  const value=summary?summary.value:(e.source || '').replace(/\s+/g,' ').slice(0,350);
  return (e.negated?'Negated fact. ':'')+(e.time?e.time+' · ':'')+value;
 }
 const n=cy.getElementById(d.id),facts=n.connectedNodes('[kind="event"]');
 const contexts=facts.slice(0,2).map(f=>description(f.data()));
 return `${n.connectedEdges().length} connections in view. `+contexts.join(' · ');
}
function observeMemory(event){
 if(document.body.dataset.view!=='graph'||graphSource==='query'||!window.CortexLive?.active())return false;
 if(!event.records?.length)return false;
 graphSource='chat';
 let limited=false;
 cy.batch(()=>{
  for(const record of event.records){
   const id='e-'+record.id;
   const data={id,label:record.frame.replaceAll('_',' '),kind:'event',live_observed:true,
    negated:!!record.negated,event:{event_id:record.id,frame:record.frame,negated:!!record.negated,
     roles:(record.bindings||[]).map(b=>({role:b.role,value:b.value})),source:''}};
   const existing=cy.getElementById(id);
   if(!existing.length){if(cy.nodes().length>=380){limited=true;continue;}cy.add({data});}else existing.data(data);
   for(const binding of record.bindings||[]){
    if(!cy.getElementById(binding.id).length){if(cy.nodes().length>=380){limited=true;continue;}cy.add({data:{id:binding.id,label:binding.value,kind:'entity',live_observed:true}});}
    const edgeId=id+'-'+binding.role;
    if(!cy.getElementById(edgeId).length)cy.add({data:{id:edgeId,source:id,target:binding.id,label:binding.role,kind:'binding'}});
   }
  }
 });
 layout();
 $('result-title').textContent='Retrieved memory';
 $('result-scope').textContent='Observed MCP results in this view · selected roles · not the generated answer';
 $('status').textContent=`${cy.nodes().length} nodes · ${cy.edges().length} recorded links · retained after activity fades${limited?' · view limit reached':''}`;
 $('sql-query').textContent='Live graph built from returned MCP records and their stored roles. No additional database query.\nRecords accumulate in this view; changing Graph workspace clears it.';
 $('page-position').textContent='';$('previous-page').disabled=true;$('next-page').disabled=true;
 return true;
}
function focusMemory(records,ids){
 if(graphSource==='query'||!window.CortexLive?.active())return false;
 graphSource='chat';
 cy.elements().remove();
 if(records.length)observeMemory({records});
 selectMemory(ids);
 $('result-title').textContent=ids.length?'Selected supporting memory':'No supporting memory selected';
 $('result-scope').textContent='Selected by the language model for the current request · not independent verification';
 $('status').textContent=`${cy.nodes().length} nodes · ${cy.edges().length} supporting links · ${records.length}/${ids.length} selected records in visual trace`;
 return true;
}
function selectMemory(ids){
 if(graphSource!=='chat')return;
 const selectedIds=new Set(ids);
 cy.nodes('[kind="event"][?live_observed]').forEach(node=>{if(!selectedIds.has(node.data('event')?.event_id))cy.remove(node);});
 cy.nodes('[kind="entity"][?live_observed]').forEach(node=>{if(node.connectedEdges().length===0)cy.remove(node);});
 layout();
 $('result-title').textContent='Prepared memory context';
 $('result-scope').textContent='Records included by the automatic retrieval gate · not certified relevance or LLM use';
 $('status').textContent=`${cy.nodes().length} nodes · ${cy.edges().length} selected-context links · ${ids.length} context records`;
}
function beginChatTurn(){
 if(document.body.dataset.view!=='graph'||!window.CortexLive?.active())return;
 request++;graphSource='chat';selected=null;pendingClarification=null;
 cy.elements().remove();layout();
 document.body.classList.remove('inspecting');$('details').replaceChildren();
 $('expand').hidden=true;$('grouping-path').hidden=true;$('clarification').hidden=true;
 $('result-title').textContent='Chat request';$('result-scope').textContent='Waiting for newly retrieved memory';
 $('status').textContent='New chat request · previous graph cleared';
 $('sql-query').textContent='Waiting for memory calls belonging to the new chat request.';
 $('query-meta').textContent='';$('page-position').textContent='';
 $('previous-page').disabled=true;$('next-page').disabled=true;
 $('apply-filter').disabled=true;$('clear-filter').disabled=true;
 $('active-filters').hidden=true;$('loading').hidden=true;$('submit-query').disabled=false;
}
window.CortexGraph={observeMemory,selectMemory,focusMemory,beginChatTurn};
function sync3D(){
 if(!space)return;
 const previous=new Map(space.graphData().nodes.map(n=>[n.id,n]));
 const nodes=cy.nodes().map(n=>{const d={...n.data()};const old=previous.get(d.id);if(old)for(const k of ['x','y','z','vx','vy','vz'])d[k]=old[k];return d;});
 const links=cy.edges().map(e=>({...e.data()}));
 space.graphData({nodes,links});
 space.onEngineStop(()=>space.zoomToFit(350,55));
}
if(!window.cytoscape){$('status').textContent='Cytoscape unavailable. Start the server and reload.';}
else {
 cy=cytoscape({headless:true,elements:[],minZoom:.15,maxZoom:3,wheelSensitivity:.2,style:[
 {selector:'node',style:{'label':'data(label)','font-family':'system-ui','font-size':11,'color':'#c5d6e8','text-wrap':'ellipsis','text-max-width':145,'text-valign':'bottom','text-margin-y':9,'width':29,'height':29,'background-color':'#8ebdff','border-width':5,'border-color':'#20334c'}},
 {selector:'node[kind="event"]',style:{'shape':'round-rectangle','width':17,'height':17,'background-color':'#88efc4','border-color':'#1e4034','font-size':9,'color':'#7fa397'}},
 {selector:'node[kind="query"]',style:{'shape':'diamond','background-color':'#c8a2f1','border-color':'#40304f','width':36,'height':36}},
 {selector:'node[?negated]',style:{'background-color':'#f1a087','border-color':'#5b332b'}},
 {selector:'edge',style:{'width':1,'line-color':'#30465a','curve-style':'bezier','label':'data(label)','font-size':8,'color':'#8197ab','text-background-color':'#0a1018','text-background-opacity':.85,'text-background-padding':2,'target-arrow-shape':'triangle','target-arrow-color':'#30465a','arrow-scale':.6}},
 {selector:'edge[kind="similarity"]',style:{'line-style':'dashed','line-color':'#9b7dbb','target-arrow-shape':'none','color':'#b49ace'}},
 {selector:'.focus',style:{'border-color':'#e0f5ed','line-color':'#8bbca9','target-arrow-color':'#8bbca9','z-index':10}},
 {selector:'node:selected',style:{'border-color':'#fff','border-width':3}}
 ]});
 cy.on('tap','node',event=>inspect(event.target));
 function submitQuestion(answer){
  if(!answer)return;
  const q=pendingClarification ? `${pendingClarification.question}\nClarification requested: ${pendingClarification.clarification}\nUser answer: ${answer}` : answer;
  load('natural',q);
 }
 $('search').addEventListener('submit',event=>{event.preventDefault();submitQuestion($('query').value.trim());});


 $('fit').onclick=()=>space.zoomToFit();$('reset').onclick=()=>{$('query').value='';pendingClarification=null;$('query').placeholder='Show, filter, group, connect…';load();};
 $('expand').onclick=()=>{if(selected && selected.data('kind')==='entity')load('expand',selected.data('label'),true);};
 $('rotate').onclick=()=>{rotating=!rotating;space.controls().autoRotate=rotating;space.controls().autoRotateSpeed=.6;$('rotate').setAttribute('aria-pressed',String(rotating));};
 new ResizeObserver(()=>{if(space){space.width($('space').clientWidth).height($('space').clientHeight);}}).observe(document.querySelector('.canvas'));
 document.addEventListener('visibilitychange',()=>{if(space){if(document.hidden || viewMode==='table')space.pauseAnimation();else space.resumeAnimation();}});
 space=canvasSpace($('space'));space.resumeAnimation();
 setupWorkbench();
 browseObjects({dataset:'memory',filters:[],offset:0});
}

function datasetFields(dataset){
 if(dataset==='memory')return currentCatalog?.frame_roles?.[browseSpec?.frame] || currentCatalog?.roles || ['entity','company','person','frame'];
 return {cases:['case_key','title','revision'],experiences:['case_key','experience_key','policy_key','action','status','reward','outcome_definition'],policies:['policy_key'],entities:['entity_key','label','scope']}[dataset] || [];
}
document.addEventListener('visibilitychange',()=>{if(!space)return;if(document.hidden)space.pauseAnimation();else if(viewMode==='graph')space.resumeAnimation();});
function setView(mode){
 const enteringGraph=mode==='graph'&&viewMode!=='graph';
 viewMode=mode;document.body.dataset.view=mode;
 $('table-view').setAttribute('aria-pressed',String(mode==='table'));
 $('graph-view').setAttribute('aria-pressed',String(mode==='graph'));
 if(enteringGraph){
  // The graph is a fresh workspace, never a visualization of an implicitly
  // selected table. Preserve table rows and browseSpec for returning to Table.
  request++;graphSource='empty';selected=null;cy.elements().remove();layout();
  document.body.classList.remove('inspecting');$('details').replaceChildren();
  $('expand').hidden=true;$('grouping-path').hidden=true;$('clarification').hidden=true;
  pendingClarification=null;$('query').placeholder='Ask a question to build the graph…';
  $('result-title').textContent='Graph';$('result-scope').textContent='No query selected';
  $('status').textContent='Graph ready · run a new query';$('page-position').textContent='';
  $('previous-page').disabled=true;$('next-page').disabled=true;
  $('apply-filter').disabled=true;$('clear-filter').disabled=true;
  $('sql-query').textContent='Run a new query to build this graph.';$('query-meta').textContent='';
  $('active-filters').hidden=true;$('loading').hidden=true;$('submit-query').disabled=false;
  document.querySelectorAll('.object-link.active').forEach(el=>el.classList.remove('active'));
  window.CortexLive?.resetView();
 }else if(mode==='table'&&lastData){
  renderRecords(lastData);showQuery(lastData);
  document.querySelectorAll('.object-link').forEach(el=>el.classList.toggle('active',el.dataset.dataset===browseSpec?.dataset&&(el.dataset.frame||'')===(browseSpec?.frame||'')));
 }
 if(space){if(mode==='table')space.pauseAnimation();else{space.width($('space').clientWidth).height($('space').clientHeight);space.resumeAnimation();}}
}

function flattenRow(row,dataset){
 if(dataset!=='memory')return row;
 const result={event_id:row.event_id,frame:row.frame};
 for(const b of row.roles || [])result[b.role]=b.value;
 if(typeof row.similarity==='number')result.similarity=row.similarity.toFixed(3);
 result.negated=row.negated;result.source=row.source;return result;
}
function displayValue(value){return value==null?'—':typeof value==='object'?JSON.stringify(value):String(value);}
function showObject(dataset,row){
 document.body.classList.add('inspecting');selected=null;$('expand').hidden=true;
 const box=$('details');box.replaceChildren();$('kind').textContent=dataset==='memory'?'Source evidence':dataset;
 text('h1',row.title || row.label || row.experience_key || row.policy_key || row.case_key || row.frame || 'Record',box);
 const dl=document.createElement('dl');box.appendChild(dl);
 for(const [key,value] of Object.entries(row)){
  if(key==='source' || key==='roles')continue;
  text('dt',key.replaceAll('_',' '),dl);text('dd',displayValue(value),dl);
 }
 if(row.roles){for(const b of row.roles){text('dt',b.role,dl);text('dd',b.value,dl);}}
 if(row.source){text('h2','Original source',box);text('pre',row.source,box);}
 text('p','Read-only view. Similarity is not evidence of identity or causation.',box).className='readonly-note';
 if(dataset==='cases' || dataset==='policies' || dataset==='memory'){
  const key=dataset==='memory'?row.event_id:dataset==='cases'?row.case_key:row.policy_key;
  if(key!=null){const ticket=++detailRequest;
   fetch('api/detail?'+new URLSearchParams({dataset,key}),{cache:'no-store'}).then(async r=>{const data=await r.json();if(!r.ok)throw Error(data.error || 'Detail unavailable');return data;}).then(data=>{
    if(ticket!==detailRequest)return;
    const source=data.event || data;
    if(dataset==='memory'){if(source.source){const pres=box.querySelectorAll('pre');if(pres.length)pres[0].textContent=source.source;else{text('h2','Original source',box);text('pre',source.source,box);}}else {box.querySelectorAll('pre').forEach(p=>p.remove());text('p','Source is inactive or unavailable in this namespace.',box);}}
    else {
     if(dataset==='cases'){
      for(const [section,items] of Object.entries(data.state || {})){if(!items.length)continue;text('h2',section.replaceAll('_',' '),box);const list=document.createElement('ul');box.appendChild(list);for(const item of items)text('li',item.text,list);}
      const summary=data.experience_summary;if(summary?.experiences?.length){text('h2','Tracked experiences',box);for(const item of summary.experiences){const card=text('div','',box);card.className='experience-card';text('strong',item.action+' · '+item.status,card);text('p',item.outcome_definition,card);if(item.observed_quote)text('p',item.observed_quote,card);if(item.reward!=null)text('small','Observed reward: '+item.reward+' · not a confidence score',card);}if(summary.has_more)text('p','More experiences available in the Experiences collection.',box);}
     }else{text('h2','Policy schema',box);text('pre',JSON.stringify(data,null,2),box);}

     const refs=new Set();function scan(v){if(!v || typeof v!=='object')return;for(const [k,x]of Object.entries(v)){if(k==='evidence_event_ids'&&Array.isArray(x))x.forEach(i=>refs.add(i));else if(k.endsWith('_event_id')&&Number.isInteger(x))refs.add(x);else if(typeof x==='object')scan(x);}}scan(data);
     if(refs.size){text('h2','Source events',box);for(const id of [...refs].slice(0,32)){const button=text('button','#'+id,box);button.className='evidence-link';button.onclick=()=>showObject('memory',{event_id:id,frame:'Source event'});}}
    }
   }).catch(error=>{if(ticket===detailRequest)text('p',error.message,box);});
  }
 }
}
function renderRecords(data){
 lastData=data;currentDataset=data.dataset || 'memory';
 if(data.plan){
  const qs=data.plan.queries || [];
  browseSpec=data.plan.mode==='relational'&&qs.length<=1?{dataset:currentDataset,frame:qs[0]?.frame || null,filters:qs[0]?.filters || [],offset:0}:null;
  if(browseSpec && currentDataset==='memory')data.next_offset=data.records_partial?50:null;
 }
 const rows=data.records || [...new Map((data.elements || []).filter(x=>x.data?.event).map(x=>[x.data.event.event_id,x.data.event])).values()];
 const flat=rows.map(row=>flattenRow(row,currentDataset));const fields=[];
 for(const row of flat)for(const k of Object.keys(row))if(!fields.includes(k)&&k!=='source_truncated')fields.push(k);
 if(fields.includes('source')){fields.splice(fields.indexOf('source'),1);fields.push('source');}
 const head=$('result-table').querySelector('thead'),body=$('result-table').querySelector('tbody');head.replaceChildren();body.replaceChildren();
 const tr=document.createElement('tr');head.appendChild(tr);for(const f of fields)text('th',f.replaceAll('_',' '),tr);
 rows.forEach((record,i)=>{const tr=document.createElement('tr');tr.tabIndex=0;body.appendChild(tr);
  for(const field of fields){const value=displayValue(flat[i][field]);const cell=text('td',value.length>200?value.slice(0,200)+'…':value,tr);cell.title=value.slice(0,1200);}
  const pick=()=>{body.querySelectorAll('tr').forEach(r=>r.classList.remove('selected'));tr.classList.add('selected');showObject(currentDataset,record);};tr.onclick=pick;tr.onkeydown=e=>{if(e.key==='Enter')pick();};
 });
 $('empty-table').hidden=rows.length>0;$('result-title').textContent=data.title || currentDataset;
 const partial=data.partial || data.records_partial;
 $('result-scope').textContent=(partial?'Partial selection · ':'')+(currentDataset==='memory'?'Source facts and semantic candidates':'APFC · stored operational objects');
 const offset=browseSpec?.offset || 0,total=data.total ?? data.matched_events;
 $('page-position').textContent=rows.length?`${offset+1}–${offset+rows.length}${total!==undefined?' of '+total:''}`:'0 records';
 $('previous-page').disabled=!browseSpec || offset===0;
 $('next-page').disabled=!browseSpec || !(data.has_more || data.records_partial);
 const select=$('filter-field'),chosen=select.value;select.replaceChildren();for(const field of datasetFields(currentDataset)){const option=text('option',field,select);option.value=field;}if([...select.options].some(o=>o.value===chosen))select.value=chosen;else if([...select.options].some(o=>o.value==='entity'))select.value='entity';
 const chips=$('active-filters');chips.replaceChildren();chips.hidden=!(browseSpec?.filters?.length);(browseSpec?.filters || []).forEach((f,i)=>{const button=text('button',f.role+' '+f.operator+' '+displayValue(f.value)+' ×',chips);button.title='Remove filter';button.onclick=()=>browseObjects({...browseSpec,filters:browseSpec.filters.filter((_,j)=>j!==i),offset:0});});
 $('apply-filter').disabled=!browseSpec;
 $('clear-filter').disabled=!browseSpec;
}
function installGraph(data){
 document.body.classList.remove('inspecting');selected=null;cy.elements().remove();
 const nodes=(data.elements || []).filter(e=>!e.data.source).slice(0,380);cy.add(nodes);
 for(const edge of data.elements || [])if(edge.data.source && !cy.getElementById(edge.data.id).length && cy.getElementById(edge.data.source).length && cy.getElementById(edge.data.target).length)cy.add(edge);
 layout();showQuery(data);renderRecords(data);
 $('status').textContent=`${data.records?.length || 0} rows loaded${data.total!==undefined?' · '+data.total+' matching records':''}${data.partial?' · paginated selection':''}`;
 $('grouping-path').hidden=true;
}
async function browseObjects(spec){
 const token=++request;pendingClarification=null;$('clarification').hidden=true;$('loading').hidden=false;$('status').textContent='Reading database…';
 $('submit-query').disabled=true;
 try{
  const params={dataset:spec.dataset,offset:String(spec.offset || 0),limit:'50',filters:JSON.stringify(spec.filters || [])};if(spec.frame)params.frame=spec.frame;
  const {response,data}=await graphRequest('api/browse?'+new URLSearchParams(params),token);
  if(!response.ok)throw Error(data.error || 'Query failed');if(token!==request)return;
  browseSpec={...spec};installGraph(data);
  document.querySelectorAll('.object-link').forEach(b=>b.classList.toggle('active',b.dataset.dataset===spec.dataset && (b.dataset.frame || '')===(spec.frame || '')));
 }catch(error){if(token===request)$('status').textContent=error.message;}
 finally{if(token===request){$('loading').hidden=true;$('submit-query').disabled=false;}}
}
async function loadCatalog(){
 try{const response=await fetch('api/catalog',{cache:'no-store'});const data=await response.json();if(!response.ok)throw Error(data.error || 'Catalog unavailable');currentCatalog=data;
  const box=$('frame-list');box.replaceChildren();for(const frame of data.frames){const button=text('button',frame.replaceAll('_',' '),box);button.className='object-link frame-link';button.dataset.dataset='memory';button.dataset.frame=frame;button.onclick=()=>browseObjects({dataset:'memory',frame,filters:[],offset:0});}
  if(lastData)renderRecords(lastData);
 }catch(error){$('frame-list').textContent=error.message;}
}
function setupWorkbench(){
 $('table-view').onclick=()=>setView('table');$('graph-view').onclick=()=>setView('graph');
 document.querySelectorAll('#navigator > .object-link').forEach(button=>{button.onclick=()=>browseObjects({dataset:button.dataset.dataset,filters:[],offset:0});});
 $('apply-filter').onclick=()=>{if(!browseSpec)return;const filter={role:$('filter-field').value,operator:$('filter-operator').value,value:$('filter-value').value};browseObjects({...browseSpec,filters:[...(browseSpec.filters || []),filter],offset:0});};
 $('filter-value').onkeydown=e=>{if(e.key==='Enter')$('apply-filter').click();};
 $('clear-filter').onclick=()=>{if(browseSpec){$('filter-value').value='';browseObjects({...browseSpec,filters:[],offset:0});}};
 $('previous-page').onclick=()=>{if(browseSpec)browseObjects({...browseSpec,offset:Math.max(0,(browseSpec.offset || 0)-50)});};
 $('next-page').onclick=()=>{if(browseSpec && lastData?.next_offset!=null)browseObjects({...browseSpec,offset:lastData.next_offset});};
 setView('table');loadCatalog();
}

}
