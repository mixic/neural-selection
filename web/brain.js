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
  let hovered = null, paused = false;
  const $ = key => document.getElementById(key);
  const role = n => n.data[2] >= 0 ? 'sensory' : n.data[1] >= 19 ? 'motor' : 'hidden';
  const color = n => n.data[6] ? '#c39bef' : role(n) === 'sensory' ? '#8dcfc4' : role(n) === 'motor' ? '#c9e870' : n.data[5] ? '#ef847c' : '#72a6d6';
  const label = n => role(n) === 'sensory' ? sensors[n.data[2]] : role(n) === 'motor' ? motors[n.data[1]-19] : `Neuron ${n.index+1}`;
  const hash = n => { let x = Math.imul(n+1, 1597334677); x ^= x >>> 16; return (x >>> 0)/4294967296; };
  const fit = () => { zoom = 1; pan = {x:0,y:0}; };
  function select(selected) {
    if (id === selected) return;
    id = selected; brain = null; nodes = []; edges = []; bySlot = new Map(); signature = ''; previousStep = null; hovered = null; tip.hidden = true; fit();
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
    const nextSignature = data.neurons.map(n=>`${n[0]}:${n[1]}:${n[2]}`).join(',')+'|'+data.synapses.map(e=>`${e[0]}:${e[1]}:${e[2]}`).join(',');
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
  function curve(e) {
    const a=e.pre,b=e.post;
    if(a===b)return [a.x,a.y,a.x-40,a.y-65,a.x+40,a.y-65,a.x,a.y];
    const dx=b.x-a.x,dy=b.y-a.y,dist=Math.hypot(dx,dy)||1;
    const bend=(hash(e.data[0])*2-1)*Math.min(60,dist*.2);
    return [a.x,a.y,a.x+dx*.33-dy/dist*bend,a.y+dy*.33+dx/dist*bend,a.x+dx*.67-dy/dist*bend,a.y+dy*.67+dx/dist*bend,b.x,b.y];
  }
  function point(c,t){const u=1-t;return [u*u*u*c[0]+3*u*u*t*c[2]+3*u*t*t*c[4]+t*t*t*c[6],u*u*u*c[1]+3*u*u*t*c[3]+3*u*t*t*c[5]+t*t*t*c[7]];}
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
      for(const e of edges) {
        const enabled=e.data[3]&&e.pre.data[4]&&e.post.data[4];
        const relevant=!focus||e.pre===focus||e.post===focus;
        const c=curve(e);ctx.beginPath();ctx.moveTo(c[0],c[1]);ctx.bezierCurveTo(...c.slice(2));
        ctx.strokeStyle=color(e.pre);ctx.globalAlpha=enabled?(relevant?.23:.035):.025;
        ctx.lineWidth=(focus&&relevant?1.7:.5)+Math.min(1.8,Math.log1p(Math.abs(e.data[4]))*.22);
        if(!enabled)ctx.setLineDash([3,5]);ctx.stroke();ctx.setLineDash([]);
        if(enabled&&relevant&&(!focus||t.s>.65)) {
          const p=point(c,.86),q=point(c,.82),angle=Math.atan2(p[1]-q[1],p[0]-q[0]);ctx.globalAlpha=.45;
          ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(p[0]-5*Math.cos(angle-.5),p[1]-5*Math.sin(angle-.5));ctx.moveTo(p[0],p[1]);ctx.lineTo(p[0]-5*Math.cos(angle+.5),p[1]-5*Math.sin(angle+.5));ctx.stroke();
        }
        const age=now-e.pre.flash;
        if(!paused&&enabled&&relevant&&age>=0&&age<900) {
          const p=point(c,age/900);ctx.globalAlpha=(1-age/900)*.9;ctx.fillStyle=color(e.pre);ctx.beginPath();ctx.arc(p[0],p[1],2.5,0,Math.PI*2);ctx.fill();
        }
      }
      ctx.globalAlpha=1;
      for(const n of nodes) {
        const c=color(n),enabled=n.data[4],age=now-n.flash,glow=!paused?Math.max(0,1-age/650):0;
        ctx.globalAlpha=enabled?1:.2;ctx.strokeStyle=c;ctx.lineWidth=.9;
        // Decorative dendritic arbor around each soma; connectivity is drawn separately above.
        for(let branch=0;branch<5;branch++) {
          const angle=branch*Math.PI*2/5+hash(n.data[0])*Math.PI;
          const len=(role(n)==='sensory'?8:12)+hash(n.data[0]+branch)*7,x=n.x+Math.cos(angle)*len,y=n.y+Math.sin(angle)*len;
          ctx.beginPath();ctx.moveTo(n.x,n.y);ctx.quadraticCurveTo(n.x+Math.cos(angle+.4)*len*.6,n.y+Math.sin(angle+.4)*len*.6,x,y);
          for(const sign of [-1,1]){ctx.moveTo(x,y);ctx.lineTo(x+Math.cos(angle+sign*.65)*6,y+Math.sin(angle+sign*.65)*6);}ctx.stroke();
        }
        if(glow>0){const g=ctx.createRadialGradient(n.x,n.y,2,n.x,n.y,24);g.addColorStop(0,`rgba(255,244,210,${glow*.8})`);g.addColorStop(1,'rgba(255,244,210,0)');ctx.fillStyle=g;ctx.beginPath();ctx.arc(n.x,n.y,24,0,Math.PI*2);ctx.fill();}
        ctx.beginPath();for(let k=0;k<12;k++){const a=k*Math.PI/6,radius=5.8+hash(n.data[0]+k)*2;const x=n.x+Math.cos(a)*radius,y=n.y+Math.sin(a)*radius;k?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.closePath();
        ctx.fillStyle=glow>.1?'#fff4d4':c;ctx.fill();ctx.strokeStyle='#e0f6ed';ctx.lineWidth=.5;ctx.stroke();
        ctx.fillStyle='#12313a';ctx.beginPath();ctx.arc(n.x+.8,n.y-.8,2.3,0,Math.PI*2);ctx.fill();
        if(n===hovered){ctx.strokeStyle='#fff4d4';ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(n.x,n.y,13,0,Math.PI*2);ctx.stroke();}
        if(n.data[1]>=0&&(role(n)!=='hidden')){ctx.font='9px system-ui';ctx.fillStyle=c;ctx.textAlign=role(n)==='motor'?'left':'right';ctx.fillText(label(n),n.x+(role(n)==='motor'?17:-17),n.y+3);}
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
    hovered=null;let best=Math.max(12,12/t.s)**2;
    for(const n of nodes){const d=(n.x-x)**2+(n.y-y)**2;if(d<best){best=d;hovered=n;}}
    tip.hidden=!hovered;if(!hovered)return;
    const n=hovered,d=n.data,mods=modulators.filter(([bit])=>d[6]&bit).map(([,name])=>name).join(', ');
    const incoming=edges.filter(e=>e.post===n&&e.data[3]&&e.pre.data[4]&&e.post.data[4]).length,outgoing=edges.filter(e=>e.pre===n&&e.data[3]&&e.pre.data[4]&&e.post.data[4]).length;
    tip.textContent=`${label(n)} · ${profiles[d[3]]||'Neuron'}${d[5]?' · inhibitory':' · excitatory'}${mods?' · '+mods:''}\n${d[4]?'Enabled':'Inactive'} · ${d[7].toFixed(1)} mV · ${n.rate.toFixed(1)} Hz\n${d[8].toLocaleString()} lifetime spikes · ${incoming} inputs / ${outgoing} outputs`;
    tip.style.whiteSpace='pre-line';tip.style.left=`${Math.max(6,Math.min(wrap.clientWidth-280,e.offsetX+18))}px`;tip.style.top=`${Math.max(6,Math.min(wrap.clientHeight-100,e.offsetY+15))}px`;
  });
  canvas.addEventListener('pointerleave',()=>{hovered=null;tip.hidden=true;});
  canvas.addEventListener('wheel',e=>{e.preventDefault();const old=transform(),x=(e.offsetX-old.x)/old.s,y=(e.offsetY-old.y)/old.s;zoom=Math.max(.6,Math.min(8,zoom*(e.deltaY<0?1.15:1/1.15)));const next=transform();pan.x+=e.offsetX-(next.x+x*next.s);pan.y+=e.offsetY-(next.y+y*next.s);},{passive:false});
  $('brain-fit').onclick=fit;
  $('brain-expand').onclick=()=>{const expanded=wrap.closest('.brain-panel').classList.toggle('expanded');$('brain-expand').textContent=expanded?'Close':'Expand';};
  document.addEventListener('keydown',e=>{if(e.code==='Escape'){wrap.closest('.brain-panel').classList.remove('expanded');$('brain-expand').textContent='Expand';}});
  requestAnimationFrame(draw);
  return {select,update};
})();
