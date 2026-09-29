/* The observer only reads snapshots. Animation never feeds back into the world. */
const brainView = (() => {
  const canvas = document.getElementById('brain');
  const ctx = canvas.getContext('2d');
  const wrap = canvas.parentElement;
  const tip = document.getElementById('brain-hover');
  const empty = document.getElementById('brain-empty');
  const sensors = ['Food A · left','Food A · ahead','Food A · right','Food B · left','Food B · ahead','Food B · right','Hunger','Fullness','Pain','Taste A','Taste B','Taste C','Animal · left','Animal · ahead','Animal · right','Taste meat','Contact A','Contact B','Contact C'];
  const motors = ['Move forward','Turn left','Turn right','Eat','Bite'];
  const profiles = ['Regular spiking','Interneuron','Lobster LP','Crab pacemaker','Aplysia LFS','Olfactory receptor'];
  const modulators = [[1,'Noradrenaline'],[2,'Serotonin'],[4,'Dopamine']];
  let brain = null, id = null, nodes = [], edges = [], bySlot = new Map();
  let signature = '', previousStep = null, zoom = 1, pan = {x:0,y:0}, drag = null;
  let hovered = null, hoveredSynapse = null, paused = false;
  const $ = key => document.getElementById(key);
  const role = n => n.data[2] >= 0 ? 'sensory' : n.data[1] >= 19 ? 'motor' : 'hidden';
  const color = n => n.data[6] ? '#c39bef' : role(n) === 'sensory' ? '#8dcfc4' : role(n) === 'motor' ? '#c9e870' : n.data[5] ? '#ef847c' : '#72a6d6';
  const label = n => role(n) === 'sensory' ? sensors[n.data[2]] : role(n) === 'motor' ? motors[n.data[1]-19] : `Neuron ${n.index+1}`;
  const hash = n => { let x = Math.imul(n+1, 1597334677); x ^= x >>> 16; return (x >>> 0)/4294967296; };
  const fit = () => { zoom = 1; pan = {x:0,y:0}; };
  function select(selected) {
    if (id === selected) return;
    id = selected; brain = null; nodes = []; edges = []; bySlot = new Map(); signature = ''; previousStep = null; hovered = null; hoveredSynapse = null; tip.hidden = true; fit();
    $('brain-title').textContent = `Brain of #${selected}`;
    $('brain-summary').textContent = 'Reading the selected network…';
    empty.hidden = false; empty.querySelector('p').textContent = 'Reading the selected network…';
  }
  function layout(data) {
    const old = bySlot;
    nodes = data.neurons.map((d,i) => ({data:d,index:i,x:old.get(d[0])?.x ?? 0,y:old.get(d[0])?.y ?? 0,flash:-Infinity,rate:0,delta:0}));
    bySlot = new Map(nodes.map(n => [n.data[0], n]));
    const groups = ['sensory','hidden','motor'].map(r => nodes.filter(n => role(n) === r).sort((a,b) => (a.data[2]-b.data[2]) || (a.data[1]-b.data[1]) || a.data[0]-b.data[0]));
    const channelOffsets = new Map();
    for (let g=0;g<3;g++) groups[g].forEach((n,i) => {
      if (old.has(n.data[0])) return;
      const channel = n.data[2], offset = channelOffsets.get(channel)||0;
      if(g===0)channelOffsets.set(channel,offset+1);
      n.x = g===0 ? 123 + offset*20 : g===2 ? 872 : 300+hash(n.data[0])*400;
      n.y = g===0 ? 55+(channel+.5)/19*450+(hash(n.data[0])-.5)*8 : g===1 ? 70+hash(n.data[0]+71)*420 : 65+(i+.5)/Math.max(1,groups[g].length)*430;
    });
    edges = data.synapses.map(d => ({data:d,pre:bySlot.get(d[1]),post:bySlot.get(d[2])})).filter(e => e.pre && e.post);
    // Layout forces are purely presentational. Recurrent and self connections remain intact.
    for (let iter=0;iter<130;iter++) {
      const forces = new Map(nodes.map(n => [n,{x:0,y:0}]));
      const grid = new Map();
      for (const n of nodes) {const key=`${Math.floor(n.x/45)},${Math.floor(n.y/45)}`; if(!grid.has(key))grid.set(key,[]);grid.get(key).push(n);}
      for (const n of nodes) {
        const gx=Math.floor(n.x/45),gy=Math.floor(n.y/45),f=forces.get(n);
        for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(const m of grid.get(`${gx+dx},${gy+dy}`)||[]) {
          if(m===n)continue;
          let x=n.x-m.x,y=n.y-m.y,d=x*x+y*y;
          if(d<1){x=hash(n.index+iter)-.5;y=hash(m.index+iter+1)-.5;d=1;}
          if(d<2000){f.x+=x*90/d;f.y+=y*90/d;}
        }
      }
      for (const e of edges) if(e.data[3] && e.pre!==e.post) {
        const x=e.post.x-e.pre.x,y=e.post.y-e.pre.y,d=Math.hypot(x,y)||1;
        const strength=(d-110)*.0009;
        forces.get(e.pre).x+=x*strength;forces.get(e.pre).y+=y*strength;
        forces.get(e.post).x-=x*strength;forces.get(e.post).y-=y*strength;
      }
      for (const n of nodes) if(role(n)==='hidden') {
        const f=forces.get(n);n.x=Math.max(225,Math.min(760,n.x+Math.max(-5,Math.min(5,f.x))));n.y=Math.max(45,Math.min(515,n.y+Math.max(-5,Math.min(5,f.y))));
      }
    }
    hovered=null;hoveredSynapse=null;tip.hidden=true;
    buildAnatomy();
  }
  function update(data, selected, stopped) {
    paused = stopped;
    if(selected===null)return;
    if(id!==selected)select(selected);
    if(!data || data.id!==id) {
      if(brain){$('brain-summary').textContent='This animal is no longer alive · final observed network';paused=true;}
      else {empty.querySelector('p').textContent='Waiting for this animal’s brain';}
      return;
    }
    const nextSignature = data.neurons.map(n=>`${n[0]}:${n[1]}:${n[2]}:${n[5]}`).join(',')+'|'+data.synapses.map(e=>`${e[0]}:${e[1]}:${e[2]}:${e[6]}`).join(',');
    const previous = new Map(nodes.map(n=>[n.data[0],{count:n.data[8],flash:n.flash,rate:n.rate,delta:n.delta}]));
    if(signature!==nextSignature){layout(data);signature=nextSignature;}
    const elapsed = previousStep===null ? 0 : (data.step-previousStep)*.016;
    const now=performance.now();
    let firing=0,spikes=0;
    for(let i=0;i<data.neurons.length;i++) {
      const d=data.neurons[i],n=bySlot.get(d[0]),old=previous.get(d[0]);n.data=d;
      if(elapsed>0 && old) {
        n.delta=Math.max(0,d[8]-old.count);n.rate=n.delta/elapsed;n.flash=n.delta ? now : old.flash;
      } else if(old) {n.flash=old.flash;n.rate=old.rate;n.delta=old.delta;}
      if(n.delta){firing++;spikes+=n.delta;}
    }
    const synapsesBySlot = new Map(data.synapses.map(d=>[d[0],d]));
    edges.forEach(e=>{e.data=synapsesBySlot.get(e.data[0]) || e.data;});
    previousStep=data.step;brain=data;empty.hidden=true;
    const active=data.neurons.filter(n=>n[4]).length;
    $('brain-summary').textContent=`${active} active neurons · ${edges.filter(e=>e.data[3]&&e.pre.data[4]&&e.post.data[4]).length} active synapses · ${firing} firing neurons / ${spikes} spikes in last sampled interval`;
  }
  function transform() { const w=canvas.clientWidth,h=canvas.clientHeight,s=Math.min(w/1000,h/560)*zoom;return {s,x:(w-1000*s)/2+pan.x,y:(h-560*s)/2+pan.y}; }
  // Morphology is a drawing of the graph: every terminal belongs to one real
  // synapse, and its receiving arbor uses that synapse's simulated branch.
  function fiber(start, end, seed, bend=1) {
    const dx=end.x-start.x,dy=end.y-start.y,d=Math.hypot(dx,dy)||1;
    const count=Math.max(2,Math.min(14,Math.ceil(d/24)));
    const points=[start];
    for(let i=1;i<count;i++) {
      const t=i/count,noise=(hash(seed+i*31)-.5)*Math.min(9,d*.08)*Math.sin(Math.PI*t)*bend;
      points.push({x:start.x+dx*t-dy/d*noise,y:start.y+dy*t+dx/d*noise});
    }
    points.push(end);return points;
  }
  function tapered(points, startWidth, endWidth) {
    for(let i=1;i<points.length;i++) {
      ctx.lineWidth=startWidth+(endWidth-startWidth)*(i-.5)/(points.length-1);
      ctx.beginPath();ctx.moveTo(points[i-1].x,points[i-1].y);ctx.lineTo(points[i].x,points[i].y);ctx.stroke();
    }
  }
  function setRoute(edge, points) {
    edge.route=points;edge.lengths=[0];
    for(let i=1;i<points.length;i++)edge.lengths.push(edge.lengths[i-1]+Math.hypot(points[i].x-points[i-1].x,points[i].y-points[i-1].y));
    edge.routeLength=edge.lengths.at(-1);
  }
  function routePoint(edge, fraction) {
    const distance=edge.routeLength*fraction;
    let low=1,high=edge.lengths.length-1;
    while(low<high){const mid=(low+high)>>1;if(edge.lengths[mid]<distance)low=mid+1;else high=mid;}
    const a=edge.route[low-1],b=edge.route[low],t=(distance-edge.lengths[low-1])/(edge.lengths[low]-edge.lengths[low-1]||1);
    return {x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t};
  }
  function buildAnatomy() {
    for(const n of nodes){n.dendrites=[];n.axons=[];n.inputs=[];n.outputs=[];}
    for(const e of edges){e.pre.outputs.push(e);e.post.inputs.push(e);}
    for(const n of nodes) {
      for(let branch=0;branch<4;branch++) {
        const inputs=n.inputs.filter(e=>e.data[6]===branch).sort((a,b)=>a.data[0]-b.data[0]);
        if(!inputs.length)continue;
        let dx=0,dy=0;
        for(const e of inputs){const x=e.pre.x-n.x,y=e.pre.y-n.y,d=Math.hypot(x,y)||1;dx+=x/d;dy+=y/d;}
        const base=Math.hypot(dx,dy)>.1?Math.atan2(dy,dx):hash(n.data[0])*Math.PI*2;
        const angle=base+(branch-1.5)*.37;
        const root={x:n.x+Math.cos(angle)*5,y:n.y+Math.sin(angle)*5};
        const trunk={x:n.x+Math.cos(angle)*19,y:n.y+Math.sin(angle)*19};
        const stem=fiber(root,trunk,n.data[0]+branch, .5);
        n.dendrites.push({points:stem,edges:inputs,start:2.9,end:1.8});
        // A binary receiving tree supplies one spine per actual incoming edge.
        function arbor(start, list, direction, depth) {
          if(list.length===1) {
            const e=list[0],a=direction+(hash(e.data[0])-.5)*.3;
            const tip={x:start.x+Math.cos(a)*12,y:start.y+Math.sin(a)*12};
            const neck={x:tip.x+Math.cos(a)*4,y:tip.y+Math.sin(a)*4};
            const head={x:neck.x+Math.cos(a)*2,y:neck.y+Math.sin(a)*2};
            e.shaftContact=Boolean(e.pre.data[5]);
            e.spine=e.shaftContact?tip:head;
            e.bouton={x:e.spine.x+Math.cos(a)*5.2,y:e.spine.y+Math.sin(a)*5.2};e.terminalAngle=a;
            n.dendrites.push({points:fiber(start,tip,e.data[0],.5),edges:list,start:1.4,end:.65});
            if(!e.shaftContact)n.dendrites.push({points:[tip,neck,head],edges:list,start:.65,end:.55});return;
          }
          const middle=Math.ceil(list.length/2);
          for(const [part,sign] of [[list.slice(0,middle),-1],[list.slice(middle),1]]) {
            if(!part.length)continue;
            const a=direction+sign*Math.max(.25,.65-depth*.07);
            const end={x:start.x+Math.cos(a)*10,y:start.y+Math.sin(a)*10};
            n.dendrites.push({points:fiber(start,end,part[0].data[0]+depth,.6),edges:part,start:Math.max(.9,1.7-depth*.18),end:Math.max(.65,1.3-depth*.16)});
            arbor(end,part,a,depth+1);
          }
        }
        arbor(trunk,inputs,angle,0);
      }
    }
    for(const n of nodes) {
      if(!n.outputs.length)continue;
      let dx=0,dy=0;
      for(const e of n.outputs){const x=e.bouton.x-n.x,y=e.bouton.y-n.y,d=Math.hypot(x,y)||1;dx+=x/d;dy+=y/d;}
      const angle=Math.hypot(dx,dy)>.1?Math.atan2(dy,dx):hash(n.data[0]+17)*Math.PI*2;
      const root={x:n.x+Math.cos(angle)*5,y:n.y+Math.sin(angle)*5};
      const junction={x:n.x+Math.cos(angle)*28,y:n.y+Math.sin(angle)*28};
      const stem=fiber(root,junction,n.data[0],.65);
      n.axons.push({points:stem,edges:n.outputs,start:2,end:1.15});
      function axonTree(start, list, route, depth) {
        if(list.length===1) {
          const e=list[0];let points;
          if(e.pre===e.post) {
            const a=angle+1.5,loop={x:n.x+Math.cos(a)*65,y:n.y+Math.sin(a)*65};
            const flank={x:n.x+Math.cos(a+1)*45,y:n.y+Math.sin(a+1)*45};
            points=[...fiber(start,loop,e.data[0]),...fiber(loop,flank,e.data[0]+1).slice(1),...fiber(flank,e.bouton,e.data[0]+2).slice(1)];
          } else points=fiber(start,e.bouton,e.data[0]);
          n.axons.push({points,edges:list,start:1.0,end:.6});
          setRoute(e,[...route,...points.slice(1)]);return;
        }
        const sorted=[...list].sort((a,b)=>Math.atan2(a.bouton.y-start.y,a.bouton.x-start.x)-Math.atan2(b.bouton.y-start.y,b.bouton.x-start.x)||a.data[0]-b.data[0]);
        const mid=Math.ceil(sorted.length/2);
        for(const part of [sorted.slice(0,mid),sorted.slice(mid)]) {
          if(!part.length)continue;
          const centroid={x:part.reduce((v,e)=>v+e.bouton.x,0)/part.length,y:part.reduce((v,e)=>v+e.bouton.y,0)/part.length};
          const x=centroid.x-start.x,y=centroid.y-start.y,d=Math.hypot(x,y)||1;
          const distance=Math.min(65,Math.max(9,d*.25));
          const end={x:start.x+x/d*distance,y:start.y+y/d*distance};
          const points=fiber(start,end,part[0].data[0]+depth,.7);
          n.axons.push({points,edges:part,start:Math.max(.75,1.2-depth*.12),end:Math.max(.6,1-depth*.12)});
          axonTree(end,part,[...route,...points.slice(1)],depth+1);
        }
      }
      axonTree(junction,n.outputs,stem,0);
    }
  }
  const connected = e => e.data[3]&&e.pre.data[4]&&e.post.data[4];
  function paintFiber(segment, owner, focus, dendrite) {
    const enabled=segment.edges.some(connected);
    const relevant=hoveredSynapse?segment.edges.includes(hoveredSynapse):!focus||owner===focus||segment.edges.some(e=>e.pre===focus||e.post===focus);
    ctx.strokeStyle=color(owner);ctx.globalAlpha=enabled?(relevant?(dendrite?.78:.44):.035):.035;
    if(!enabled)ctx.setLineDash([2,4]);
    const scale=focus&&relevant?1.25:1;
    tapered(segment.points,segment.start*scale,segment.end*scale);ctx.setLineDash([]);
  }
  function draw(now) {
    const w=canvas.clientWidth,h=canvas.clientHeight,r=Math.min(devicePixelRatio||1,2);
    if(canvas.width!==Math.round(w*r)||canvas.height!==Math.round(h*r)){canvas.width=Math.round(w*r);canvas.height=Math.round(h*r);}
    ctx.setTransform(r,0,0,r,0,0);ctx.clearRect(0,0,w,h);
    const bg=ctx.createRadialGradient(w*.5,h*.5,0,w*.5,h*.5,w*.6);bg.addColorStop(0,'#172c32');bg.addColorStop(1,'#0b1519');ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
    if(brain) {
      const t=transform();ctx.save();ctx.translate(t.x,t.y);ctx.scale(t.s,t.s);
      ctx.font='10px system-ui';ctx.fillStyle='#799a9d';ctx.textAlign='center';
      ctx.fillText('SENSORY INPUT',112,24);ctx.fillText('RECURRENT NETWORK',495,24);ctx.fillText('MOTOR OUTPUT',875,24);
      const focus=hovered;
      ctx.lineCap='round';ctx.lineJoin='round';
      // Shared outgoing trunks bifurcate into real presynaptic terminals.
      for(const n of nodes)for(const segment of n.axons)paintFiber(segment,n,focus,false);
      // The receiving tree is continuous from every spine back to its soma.
      for(const n of nodes)for(const segment of n.dendrites)paintFiber(segment,n,focus,true);
      for(const e of edges) {
        const enabled=connected(e),relevant=hoveredSynapse?e===hoveredSynapse:!focus||e.pre===focus||e.post===focus;
        const age=now-e.pre.flash,glow=!paused?Math.max(0,1-age/900):0;
        ctx.globalAlpha=enabled?(relevant?.9:.08):.07;
        // Bouton and spine head face each other across a visible synaptic gap.
        if(!e.shaftContact){ctx.fillStyle=color(e.post);ctx.beginPath();ctx.arc(e.spine.x,e.spine.y,1.4,0,Math.PI*2);ctx.fill();}
        ctx.save();ctx.translate(e.bouton.x,e.bouton.y);ctx.rotate(e.terminalAngle);
        ctx.fillStyle=glow&&enabled&&relevant?'#fff4d4':color(e.pre);
        ctx.beginPath();ctx.ellipse(0,0,2.3,1.65,0,0,Math.PI*2);ctx.fill();ctx.restore();
        if(!paused&&enabled&&relevant&&age>=0&&age<900) {
          const p=routePoint(e,age/900);ctx.globalAlpha=(1-age/900)*.9;
          ctx.fillStyle='#fff4d4';ctx.beginPath();ctx.arc(p.x,p.y,2.1,0,Math.PI*2);ctx.fill();
        }
      }
      ctx.globalAlpha=1;
      for(const n of nodes) {
        const c=color(n),enabled=n.data[4],age=now-n.flash,glow=!paused?Math.max(0,1-age/650):0;
        ctx.globalAlpha=enabled?1:.2;
        if(glow>0){const g=ctx.createRadialGradient(n.x,n.y,2,n.x,n.y,24);g.addColorStop(0,`rgba(255,244,210,${glow*.8})`);g.addColorStop(1,'rgba(255,244,210,0)');ctx.fillStyle=g;ctx.beginPath();ctx.arc(n.x,n.y,24,0,Math.PI*2);ctx.fill();}
        ctx.beginPath();for(let k=0;k<12;k++){const a=k*Math.PI/6,radius=5.8+hash(n.data[0]+k)*2;const x=n.x+Math.cos(a)*radius,y=n.y+Math.sin(a)*radius;k?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.closePath();
        ctx.fillStyle=glow>.1?'#fff4d4':c;ctx.fill();ctx.strokeStyle='#e0f6ed';ctx.lineWidth=.5;ctx.stroke();
        ctx.fillStyle='#12313a';ctx.beginPath();ctx.arc(n.x+.8,n.y-.8,2.3,0,Math.PI*2);ctx.fill();
        if(n===hovered||n===hoveredSynapse?.pre||n===hoveredSynapse?.post){ctx.strokeStyle='#fff4d4';ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(n.x,n.y,13,0,Math.PI*2);ctx.stroke();}
        if(n.data[1]>=0&&(role(n)!=='hidden')){ctx.font='9px system-ui';ctx.fillStyle=c;ctx.textAlign=role(n)==='motor'?'left':'right';ctx.fillText(label(n),role(n)==='motor'?n.x+17:96,n.y+3);}
      }
      ctx.globalAlpha=1;ctx.restore();
    }
    requestAnimationFrame(draw);
  }
  canvas.addEventListener('pointerdown',e=>{drag={x:e.offsetX,y:e.offsetY,px:pan.x,py:pan.y};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointerup',()=>{drag=null;});
  canvas.addEventListener('pointercancel',()=>{drag=null;});
  canvas.addEventListener('pointermove',e=>{
    if(drag&&e.buttons){pan.x=drag.px+e.offsetX-drag.x;pan.y=drag.py+e.offsetY-drag.y;tip.hidden=true;return;}
    const t=transform(),x=(e.offsetX-t.x)/t.s,y=(e.offsetY-t.y)/t.s;
    hovered=null;hoveredSynapse=null;let best=Math.max(10,8/t.s)**2;
    for(const n of nodes){const d=(n.x-x)**2+(n.y-y)**2;if(d<best){best=d;hovered=n;}}
    if(!hovered)for(const e of edges){const d=(e.bouton.x-x)**2+(e.bouton.y-y)**2;if(d<best){best=d;hoveredSynapse=e;}}
    tip.hidden=!hovered&&!hoveredSynapse;if(tip.hidden)return;
    if(hoveredSynapse) {
      const e=hoveredSynapse;
      tip.textContent=`Synapse ${e.data[0]} · ${connected(e)?'Enabled':'Inactive'}\n${label(e.pre)} → ${label(e.post)}\nDendrite ${e.data[6]+1} · delay ${e.data[5]} ms · weight ${e.data[4].toFixed(2)} nS${e.data[7]?' · plastic':''}\nBouton → synaptic gap → ${e.shaftContact?'dendritic shaft':'spine'}`;
    } else {
    const n=hovered,d=n.data,mods=modulators.filter(([bit])=>d[6]&bit).map(([,name])=>name).join(', ');
    const incoming=edges.filter(e=>e.post===n&&e.data[3]&&e.pre.data[4]&&e.post.data[4]).length,outgoing=edges.filter(e=>e.pre===n&&e.data[3]&&e.pre.data[4]&&e.post.data[4]).length;
    tip.textContent=`${label(n)} · ${profiles[d[3]]||'Neuron'}${d[5]?' · inhibitory':' · excitatory'}${mods?' · '+mods:''}\n${d[4]?'Enabled':'Inactive'} · ${d[7].toFixed(1)} mV · ${n.rate.toFixed(1)} Hz\n${d[8].toLocaleString()} lifetime spikes · ${incoming} inputs / ${outgoing} outputs`;
    }
    tip.style.whiteSpace='pre-line';tip.style.left=`${Math.max(6,Math.min(wrap.clientWidth-280,e.offsetX+18))}px`;tip.style.top=`${Math.max(6,Math.min(wrap.clientHeight-100,e.offsetY+15))}px`;
  });
  canvas.addEventListener('pointerleave',()=>{hovered=null;hoveredSynapse=null;tip.hidden=true;});
  canvas.addEventListener('wheel',e=>{e.preventDefault();const old=transform(),x=(e.offsetX-old.x)/old.s,y=(e.offsetY-old.y)/old.s;zoom=Math.max(.6,Math.min(8,zoom*(e.deltaY<0?1.15:1/1.15)));const next=transform();pan.x+=e.offsetX-(next.x+x*next.s);pan.y+=e.offsetY-(next.y+y*next.s);},{passive:false});
  $('brain-fit').onclick=fit;
  $('brain-expand').onclick=()=>{const expanded=wrap.closest('.brain-panel').classList.toggle('expanded');$('brain-expand').textContent=expanded?'Close':'Expand';};
  document.addEventListener('keydown',e=>{if(e.code==='Escape'){wrap.closest('.brain-panel').classList.remove('expanded');$('brain-expand').textContent='Expand';}});
  requestAnimationFrame(draw);
  return {select,update};
})();
