'use strict';
// Only real MCP memory events. No inferred neural activity or simulated traffic.
window.CortexLive=(()=>{
 let enabled=true, timer=null, abort=null, cursor=0, generation=null, epoch=0;
 const client=crypto.randomUUID().replaceAll('-','');
 const bursts=[],history=[];
 let building=false,activeTurn=null,lastModel=null;
 const candidateCache=new Map();
 const activeSkills=new Set();
 function clearSkills(){activeSkills.clear();$('skill-state').hidden=true;}
 let queryPackets=0,droppedQuery=0;
 let packets=0, overflow=0, rateDrops=0, lastPoll=0;
 const $=id=>document.getElementById(id);
 const active=()=>enabled && !document.hidden && document.body.dataset.view==='graph';
 const hue=event=>event.phase==='error'?'255,96,103':event.operation==='memory_focus'?'140,235,180':!event.pipeline?'130,145,165':/group_join|deduplicate/.test(event.operation)?'194,152,255':/remember|update|complete_experience|deactivate|forget|restore/.test(event.operation)?'255,196,100':'115,222,255';
 function status(message){$('live-state').textContent=message;}
 function trim(now){while(bursts.length && now-bursts[0].born>6500)bursts.shift();}
 function ingest(events){
  const boundary=events.findLastIndex(e=>e.origin!=='viewer'&&e.operation==='chat_turn'&&e.phase==='start');
  if(boundary>=0){
   window.CortexGraph?.beginChatTurn();
   resetView();activeTurn=events[boundary].call;candidateCache.clear();
   lastModel={name:events[boundary].model||null,engine:events[boundary].engine||null,effort:events[boundary].effort||null,producer:events[boundary].producer,seen:new Date().toISOString()};
   window.dispatchEvent(new Event('cortex-model-info'));
   events=events.slice(boundary);
  }
  const now=performance.now();
  for(const event of events){
   if(event.origin==='viewer')continue;
   if(event.operation==='execution_trace'&&event.request_id===activeTurn)window.CortexGraph?.observeExecution(event);
   if(event.operation==='skill_bypass'){
    if(event.phase==='start')activeSkills.add(event.call);else activeSkills.delete(event.call);
    $('skill-state').hidden=!activeSkills.size;
   }
   for(const record of event.records||[]){candidateCache.set(record.id,record);if(candidateCache.size>192)candidateCache.delete(candidateCache.keys().next().value);}
  }
  const focus=events.filter(e=>e.operation==='memory_focus'&&e.request_id===activeTurn).at(-1);
  let retained=false,focusRecords=[];
  if(focus){
   focusRecords=(focus.selected_ids||[]).map(id=>candidateCache.get(id)).filter(Boolean);
   retained=window.CortexGraph?.focusMemory(focusRecords,focus.selected_ids||[]);
   for(const burst of bursts)if(!burst.pipeline)burst.collapseAt=now;
  }
  for(const event of events){
   if(event.operation==='memory_focus'&&event!==focus)continue;
   if(!['start','complete','error'].includes(event.phase) || event.origin==='viewer')continue;
   packets++;
   const focusedEvent=event===focus;
   const eventRecords=focusedEvent?focusRecords:(event.records||[]);
   const nodes=eventRecords.flatMap(r=>['e-'+r.id,...(r.bindings||[]).map(b=>b.id)]);
   const joins=eventRecords.flatMap(r=>(r.bindings||[]).map(b=>({source:'e-'+r.id,target:b.id})));
   if(event.operation==='execution_trace')continue;
   bursts.push({...event,born:now,...(retained&&focusedEvent?{pipeline:true,nodes,joins}:{collapseAt:focus?now:undefined})});
   if(bursts.length>48)bursts.shift();
   const label=event.phase==='start'?'started':event.phase==='error'?'failed':`${event.count} matched · ${event.duration_ms.toFixed(1)} ms`;
   history.unshift(`${event.origin==='gate'?(event.operation==='memory_selection'?'Prepared context':'Exploration'):'Client '+event.producer} · ${event.operation} · ${label}`);
  }
  history.splice(8);
  $('live-events').replaceChildren(...history.map(line=>{const el=document.createElement('div');el.textContent=line;return el;}));
  $('live-metrics').textContent=`${queryPackets} query transitions · ${packets} MCP events · ${droppedQuery} skipped notices`;
  if(events.some(e=>e.origin!=='viewer')&&!building)status('Live · observed memory operations');
 }
 async function poll(mine){
  if(!active()||mine!==epoch)return;
  abort=new AbortController();
  try{
   const response=await fetch('api/live?'+new URLSearchParams({client,cursor:String(cursor)}),{cache:'no-store',signal:abort.signal});
   if(!response.ok)throw Error('Live channel unavailable');
   const data=await response.json();
   if(!active()||mine!==epoch)return;
   if(generation && generation!==data.generation){cursor=0;generation=data.generation;bursts.length=0;history.length=0;}
   generation=data.generation;cursor=data.cursor;overflow=data.overflow||0;rateDrops=data.rate_drops||0;
   ingest(data.events||[]);lastPoll=performance.now();
   if(!(data.events||[]).length&&!building&&!bursts.length)status('Live · waiting for memory activity');
  }catch(error){if(error.name!=='AbortError')status('Live disconnected · retrying');}
  finally{abort=null;if(active()&&mine===epoch)timer=setTimeout(()=>poll(mine),1000);}
 }
 function stop(){
  clearSkills();
  epoch++;clearTimeout(timer);timer=null;
  if(abort){abort.abort();abort=null;}
  fetch('api/live?'+new URLSearchParams({client,stop:'1'}),{cache:'no-store',keepalive:true}).catch(()=>{});
  bursts.length=0;status(enabled?'Live paused · graph not visible':'Static · live collection off');
 }
 function reconcile(){
  $('live-toggle').checked=enabled;
  $('live-panel').hidden=!enabled||document.body.dataset.view!=='graph';
  if(active()){if(!timer&&!abort){epoch++;poll(epoch);}}
  else if(timer||abort||bursts.length)stop();
 }
 function toggle(){
  enabled=!enabled;

  if(!enabled)stop();
  reconcile();
 }
 function resetView(){clearSkills();activeTurn=null;candidateCache.clear();bursts.length=0;history.length=0;building=false;queryPackets=0;droppedQuery=0;$('live-events').replaceChildren();status('Graph ready · waiting for a new query or memory activity');}
 function beginQuery(){building=true;queryPackets=0;droppedQuery=0;bursts.length=0;history.length=0;$('live-events').replaceChildren();status('Live · query executing');}
 function endQuery(dropped=0){building=false;droppedQuery=dropped;status(dropped?`Query complete · ${dropped} visual notices skipped · final result preserved`:'Query complete · flares settle into the static result');}
 function pipeline(event){
  if(!active())return;
  queryPackets++;$('live-metrics').textContent=`${queryPackets} query transitions · ${packets} MCP events`;
  const names={planning:'Planning query',plan_ready:'Validated plan',retrieve_start:'Reading memory',retrieved:'Records returned',classify:'Assigning stored grouping values',deduplicate:'Merging duplicate identity',bind:'Binding recorded roles',group_join:'Joining group membership',object_join:'Joining recorded object links'};
  status(names[event.stage]||event.stage);
  const elements=event.elements||[];
  const nodes=elements.filter(e=>!e.data.source).map(e=>e.data.id);
  const joins=elements.filter(e=>e.data.source).map(e=>({source:e.data.source,target:e.data.target}));
  bursts.push({born:performance.now(),phase:'complete',operation:event.stage,nodes,joins,pipeline:true,records:[]});
  if(bursts.length>48)bursts.shift();
  history.unshift(`${names[event.stage]||event.stage} · ${event.elapsed_ms.toFixed(1)} ms${joins.length?' · '+joins.length+' executed links':''}`);
  history.splice(8);$('live-events').replaceChildren(...history.map(line=>{const el=document.createElement('div');el.textContent=line;return el;}));
 }
 function hash(value){let h=2166136261;for(const ch of value){h^=ch.charCodeAt(0);h=Math.imul(h,16777619);}return h>>>0;}
 function render(ctx,points,w,h,routes=new Map()){
  if(!active())return;
  const now=performance.now();trim(now);
  const byId=new Map(points.map(p=>[p.n.id,p]));
  const hub={x:w*.5,y:h*.79};
  ctx.save();
  const maxNodes=80;let used=0;
  const seenNodes=new Set(),seenJoins=new Set();
  for(const event of [...bursts].reverse()){
   const age=(now-event.born)/1000,collapse=event.collapseAt?Math.max(0,1-(now-event.collapseAt)/800):1,fade=Math.max(0,1-age/6.5)*collapse,rgb=hue(event);
   if(collapse<=0)continue;
   ctx.globalAlpha=fade;
   if(event.pipeline){
    for(const join of event.joins){
     const key=join.source+'\0'+join.target;if(seenJoins.has(key)||seenJoins.size>=160)continue;seenJoins.add(key);
     const a=byId.get(join.source),b=byId.get(join.target);if(!a||!b)continue;
     // Motion is constrained to an actual executed edge, in its stored direction.
     const route=routes.get(key)||[a,b];
     const lengths=[0];for(let i=1;i<route.length;i++)lengths.push(lengths[i-1]+Math.hypot(route[i].x-route[i-1].x,route[i].y-route[i-1].y));
     const total=lengths.at(-1)||1;
     const at=q=>{const distance=Math.max(0,Math.min(1,q))*total;let i=1;while(i<lengths.length-1&&lengths[i]<distance)i++;const f=(distance-lengths[i-1])/(lengths[i]-lengths[i-1]||1);return {x:route[i-1].x+(route[i].x-route[i-1].x)*f,y:route[i-1].y+(route[i].y-route[i-1].y)*f};};
     ctx.strokeStyle=`rgba(${rgb},${fade*.6})`;ctx.lineWidth=2;
     ctx.beginPath();route.forEach((p,i)=>{if(i)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y);});ctx.stroke();
     if(age<1.1){
      const t=Math.min(1,age/1.1);
      for(let j=7;j>=0;j--){const p0=at(t-(j+1)*.025),p1=at(t-j*.025);ctx.strokeStyle=`rgba(${rgb},${fade*(1-j/8)*.9})`;ctx.lineWidth=1+2*(1-j/8);ctx.beginPath();ctx.moveTo(p0.x,p0.y);ctx.lineTo(p1.x,p1.y);ctx.stroke();}
      const tip=at(t);ctx.fillStyle=`rgb(${rgb})`;ctx.shadowColor=`rgb(${rgb})`;ctx.shadowBlur=12;ctx.beginPath();ctx.arc(tip.x,tip.y,3,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;}
    }
    for(const id of event.nodes){
     if(seenNodes.has(id)||seenNodes.size>=120)continue;seenNodes.add(id);
     const point=byId.get(id);if(!point)continue;
     const amount=point.n.kind==='group'?Math.log2(1+(point.n.members_count||1)):1;
     const radius=Math.min(52,point.r+4+Math.sin(Math.min(1,age/3)*Math.PI)*(8+amount*3));
     const glow=ctx.createRadialGradient(point.x,point.y,0,point.x,point.y,radius*2);
     glow.addColorStop(0,`rgba(${rgb},${fade*.38})`);glow.addColorStop(1,`rgba(${rgb},0)`);ctx.fillStyle=glow;
     ctx.beginPath();ctx.arc(point.x,point.y,radius*2,0,Math.PI*2);ctx.fill();
    }
    continue;
   }
   if(event.phase==='start'){
    const radius=12+age*18;
    ctx.strokeStyle=`rgba(${rgb},${fade*.5})`;ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(hub.x,hub.y,radius,0,Math.PI*2);ctx.stroke();
    continue;
   }
   for(const record of event.records||[]){
    if(++used>maxNodes)break;
    const known=byId.get('e-'+record.id) || points.find(p=>p.n.event?.event_id===record.id);
    const key=hash(record.frame),a=(key%6283)/1000;
    const center={x:w*.5+Math.cos(a)*w*.28,y:h*.44+Math.sin(a)*h*.24};
    const b=(hash(String(record.id))%6283)/1000;
    const target=known||{x:center.x+Math.cos(b)*30,y:center.y+Math.sin(b)*30,r:4};
    // A direct path means this operation returned this record, not a causal inference.
    ctx.strokeStyle=`rgba(${rgb},${fade*.25})`;ctx.lineWidth=1;ctx.setLineDash([3,5]);
    ctx.beginPath();ctx.moveTo(hub.x,hub.y);ctx.lineTo(target.x,target.y);ctx.stroke();ctx.setLineDash([]);
    if(age<1.5){const t=Math.min(1,age/1.5);ctx.fillStyle=`rgb(${rgb})`;ctx.shadowColor=`rgb(${rgb})`;ctx.shadowBlur=10;
     ctx.beginPath();ctx.arc(hub.x+(target.x-hub.x)*t,hub.y+(target.y-hub.y)*t,2.7,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;}
    const pulse=Math.max(0,1-age/6.5)*collapse,radius=((target.r||4)+8+Math.sin(Math.min(1,age/3)*Math.PI)*18)*collapse;
    const glow=ctx.createRadialGradient(target.x,target.y,0,target.x,target.y,radius*2);
    glow.addColorStop(0,`rgba(${rgb},${pulse*.55})`);glow.addColorStop(1,`rgba(${rgb},0)`);
    ctx.fillStyle=glow;ctx.beginPath();ctx.arc(target.x,target.y,radius*2,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=`rgba(${rgb},${pulse*.7})`;ctx.beginPath();ctx.arc(target.x,target.y,radius,0,Math.PI*2);ctx.stroke();
    if(!known){ctx.fillStyle=`rgba(${rgb},${fade})`;ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillText(`${record.bindings?.find(b=>b.role==='entity')?.value||record.frame} · #${record.id}`,target.x,target.y+radius+12);}
   }
  }
  if(bursts.some(e=>!e.pipeline)){ctx.globalAlpha=1;ctx.fillStyle='#cdefff';ctx.font='11px system-ui';ctx.textAlign='center';ctx.fillText('Memory operations',hub.x,hub.y+20);}
  ctx.restore();
 }
 function install(){
  $('live-toggle').addEventListener('change',toggle);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();else reconcile();});
  window.addEventListener('pagehide',stop);
  reconcile();
  new MutationObserver(reconcile).observe(document.body,{attributes:true,attributeFilter:['data-view']});
 }
 document.addEventListener('DOMContentLoaded',install);
 return {modelInfo:()=>lastModel,render,active,resetView,beginQuery,endQuery,pipeline,animating:()=>active()&&bursts.length>0};
})();
