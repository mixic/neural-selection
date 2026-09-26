const $ = id => document.getElementById(id);
const canvas = $('world');
const ctx = canvas.getContext('2d');
const chart = $('population-chart');
const chartCtx = chart.getContext('2d');
const wrap = $('map-wrap');
const foodLayer = document.createElement('canvas');
const foodCtx = foodLayer.getContext('2d');
const colors = ['#000000', '#64b788', '#e5a577', '#78a8d8'];
let state = null;
let selectedId = null;
let lastKnown = null;
let events = [];
let history = [];
let zoom = 1;
let panX = 0;
let panY = 0;
let drag = null;
let hoverAnimal = null;
const fmt = n => Number(n).toLocaleString('en-GB');

function sizeCanvas(node, context) {
  const rect = node.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (node.width !== width || node.height !== height) {
    node.width = width;
    node.height = height;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {width: rect.width, height: rect.height, ratio};
}

function transform() {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const worldWidth = state?.width || 128, worldHeight = state?.height || 128;
  const scale = Math.min(width / worldWidth, height / worldHeight) * zoom;
  return {scale, x: (width - worldWidth * scale) / 2 + panX,
    y: (height - worldHeight * scale) / 2 + panY};
}

function worldPoint(x, y) {
  const t = transform();
  return {x: t.x + x * t.scale, y: t.y + y * t.scale};
}

function rebuildFood(data) {
  const width = data.width, height = data.height;
  foodLayer.width = width;
  foodLayer.height = height;
  const pixels = foodCtx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const kind = data.food.charCodeAt(i) - 48;
    if (kind < 1 || kind > 3) continue;
    const color = colors[kind];
    const offset = i * 4;
    pixels.data[offset] = parseInt(color.slice(1, 3), 16);
    pixels.data[offset + 1] = parseInt(color.slice(3, 5), 16);
    pixels.data[offset + 2] = parseInt(color.slice(5, 7), 16);
    pixels.data[offset + 3] = 215;
  }
  foodCtx.putImageData(pixels, 0, 0);
}

function drawWorld(now) {
  const {width, height} = sizeCanvas(canvas, ctx);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0d1719';
  ctx.fillRect(0, 0, width, height);
  if (!state) {
    ctx.fillStyle = '#91aaa4';
    ctx.font = '13px DM Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for world data…', width / 2, height / 2);
    requestAnimationFrame(drawWorld);
    return;
  }
  const t = transform();
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, width, height);
  ctx.clip();
  ctx.fillStyle = '#152729';
  ctx.fillRect(t.x, t.y, state.width * t.scale, state.height * t.scale);
  if (t.scale >= 10) {
    ctx.strokeStyle = '#ffffff09';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= state.width; x += 8) {
      ctx.moveTo(t.x + x * t.scale, t.y);
      ctx.lineTo(t.x + x * t.scale, t.y + state.height * t.scale);
    }
    for (let y = 0; y <= state.height; y += 8) {
      ctx.moveTo(t.x, t.y + y * t.scale);
      ctx.lineTo(t.x + state.width * t.scale, t.y + y * t.scale);
    }
    ctx.stroke();
  }
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(foodLayer, t.x, t.y, state.width * t.scale, state.height * t.scale);
  const marker = Math.max(3, Math.min(6, t.scale * 0.75));
  for (const animal of state.animals) {
    const x = t.x + animal[1] * t.scale;
    const y = t.y + animal[2] * t.scale;
    if (x < -15 || y < -15 || x > width + 15 || y > height + 15) continue;
    const chosen = animal[0] === selectedId;
    if (chosen) {
      ctx.beginPath();
      ctx.arc(x, y, marker + 7, 0, Math.PI * 2);
      ctx.strokeStyle = '#c9e870';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(animal[3]);
    ctx.beginPath();
    ctx.moveTo(marker * 1.35, 0);
    ctx.lineTo(-marker, -marker * 0.76);
    ctx.lineTo(-marker * 0.55, 0);
    ctx.lineTo(-marker, marker * 0.76);
    ctx.closePath();
    ctx.fillStyle = chosen ? '#d8f08c' : '#f0f1dc';
    ctx.fill();
    ctx.restore();
  }
  events = events.filter(event => now - event.time < 1300);
  for (const event of events) {
    const age = (now - event.time) / 1300;
    const x = t.x + event.x * t.scale, y = t.y + event.y * t.scale;
    if (x < -20 || y < -20 || x > width + 20 || y > height + 20) continue;
    ctx.globalAlpha = 1 - age;
    ctx.strokeStyle = event.type === 'birth' ? '#8dcfc4' : '#ef847c';
    ctx.lineWidth = 1.5;
    if (event.type === 'birth') {
      ctx.beginPath();
      ctx.arc(x, y, 4 + age * 14, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      const r = 3 + age * 6;
      ctx.beginPath();
      ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
      ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = '#71928a';
  ctx.lineWidth = 1;
  ctx.strokeRect(t.x + .5, t.y + .5, state.width * t.scale, state.height * t.scale);
  ctx.restore();
  requestAnimationFrame(drawWorld);
}

function drawChart() {
  const {width, height} = sizeCanvas(chart, chartCtx);
  chartCtx.clearRect(0, 0, width, height);
  if (history.length < 2) return;
  const max = Math.max(10, ...history.map(x => x.population)) * 1.15;
  const minStep = history[0].step;
  const range = Math.max(1, history.at(-1).step - minStep);
  const points = history.map(h => [4 + (h.step - minStep) / range * (width - 8),
    height - 5 - h.population / max * (height - 14)]);
  const fill = chartCtx.createLinearGradient(0, 0, 0, height);
  fill.addColorStop(0, '#c9e87055');
  fill.addColorStop(1, '#c9e87000');
  chartCtx.beginPath();
  chartCtx.moveTo(points[0][0], height);
  for (const [x, y] of points) chartCtx.lineTo(x, y);
  chartCtx.lineTo(points.at(-1)[0], height);
  chartCtx.closePath();
  chartCtx.fillStyle = fill;
  chartCtx.fill();
  chartCtx.beginPath();
  points.forEach(([x, y], i) => i ? chartCtx.lineTo(x, y) : chartCtx.moveTo(x, y));
  chartCtx.strokeStyle = '#c9e870';
  chartCtx.lineWidth = 2;
  chartCtx.stroke();
  chartCtx.beginPath();
  chartCtx.arc(...points.at(-1), 3, 0, Math.PI * 2);
  chartCtx.fillStyle = '#dff3a2';
  chartCtx.fill();
}

function updateInspector() {
  $('inspector-empty').hidden = selectedId !== null;
  $('inspector-detail').hidden = selectedId === null;
  if (selectedId === null || !state) return;
  const animal = state.animals.find(a => a[0] === selectedId);
  if (animal) lastKnown = animal;
  if (!lastKnown) return;
  const a = lastKnown;
  $('animal-id').textContent = `#${a[0]}`;
  $('animal-generation').textContent = `GEN ${a[7]}`;
  $('animal-presence').textContent = animal ? '● Alive and moving' : '● No longer alive';
  $('animal-presence').classList.toggle('lost', !animal);
  $('energy').textContent = a[4].toFixed(1);
  $('health').textContent = a[5].toFixed(1);
  $('energy-bar').style.width = `${Math.max(0, Math.min(100, a[4] / 22 * 100))}%`;
  $('health-bar').style.width = `${Math.max(0, Math.min(100, a[5] / 10 * 100))}%`;
  $('age').textContent = `${fmt(a[6])} steps`;
  $('position').textContent = `${a[1].toFixed(0)}, ${a[2].toFixed(0)}`;
  $('neurons').textContent = fmt(a[8]);
  $('synapses').textContent = fmt(a[9]);
  $('active-neurons').textContent = fmt(a[10]);
  $('active-synapses').textContent = fmt(a[11]);
  $('fullness').textContent = a[12].toFixed(1);
  $('pain').textContent = a[13].toFixed(1);
  $('taste-bars').innerHTML = ['A', 'B', 'C', 'Meat'].map((name, i) =>
    `<div class="taste-row"><label>${name}</label><div><i style="width:${Math.min(100, Math.max(0, a[14 + i] * 100))}%"></i></div><em>${a[14 + i].toFixed(1)}</em></div>`).join('');
}

function updateUI(data) {
  const status = data.finished ? 'RUN COMPLETE' : data.paused ? 'PAUSED' : 'LIVE';
  $('status').textContent = status;
  $('status-dot').className = 'status-dot ' + (data.finished ? 'finished' : data.paused ? 'paused' : '');
  $('step-label').textContent = data.total > 0
    ? `STEP ${fmt(data.step)} / ${fmt(data.total)}`
    : `STEP ${fmt(data.step)} · NO LIMIT`;
  $('population').textContent = fmt(data.population);
  for (const [id, field] of [['births','births'], ['deaths','deaths'],
    ['predation','predation'], ['bites','bites'], ['food-cells','food_cells'],
    ['generation','max_generation']]) $(id).textContent = fmt(data[field]);
  $('pause').textContent = data.paused ? 'Resume' : 'Pause';
  $('pause').disabled = data.finished;
  $('end').disabled = data.finished;
  for (const speed of ['1', '4', '0'])
    $(`speed${speed}`).classList.toggle('active', data.speed === Number(speed));
  const foodNames = ['A · nutritious', 'B · harmful', 'C · inert'];
  $('choice-rows').innerHTML = (data.food_choice || []).map((item, i) => {
    const accepted = item[0], opportunities = item[1];
    const share = opportunities ? accepted / opportunities : 0;
    const alignment = item[3] ? item[2] / item[3] : 0;
    return `<div class="choice-row"><div><span>${foodNames[i]}</span><strong>${opportunities ? (share * 100).toFixed(1) + '%' : '—'}</strong></div><div class="choice-track"><i style="width:${(share * 100).toFixed(1)}%"></i></div><small>${fmt(opportunities)} encounters · approach ${alignment >= 0 ? '+' : ''}${alignment.toFixed(2)}</small></div>`;
  }).join('');
  updateInspector();
}

async function poll() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!Number.isFinite(data.step)) { setTimeout(poll, 150); return; }
    if (data.step !== state?.step) {
      if (state) {
        const old = new Map(state.animals.map(a => [a[0], a]));
        const current = new Set(data.animals.map(a => a[0]));
        const now = performance.now();
        for (const a of data.animals) if (!old.has(a[0]))
          events.push({type: 'birth', x: a[1], y: a[2], time: now});
        for (const a of state.animals) if (!current.has(a[0]))
          events.push({type: 'death', x: a[1], y: a[2], time: now});
      }
      state = data;
      rebuildFood(data);
      history.push({step: data.step, population: data.population});
      if (history.length > 250) history = history.filter((_, i) => i % 2 === 0);
      drawChart();
    } else if (state) {
      state = data;
    }
    updateUI(data);
  } catch (error) {
    $('status').textContent = 'DISCONNECTED';
    $('status-dot').className = 'status-dot offline';
  }
  setTimeout(poll, 150);
}

function nearestAnimal(x, y) {
  if (!state) return null;
  const t = transform();
  let nearest = null, distance = 14 * 14;
  for (const a of state.animals) {
    const dx = x - (t.x + a[1] * t.scale);
    const dy = y - (t.y + a[2] * t.scale);
    const d = dx * dx + dy * dy;
    if (d < distance) { distance = d; nearest = a; }
  }
  return nearest;
}

canvas.addEventListener('pointerdown', event => {
  drag = {x: event.offsetX, y: event.offsetY, panX, panY, moved: false};
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (drag && event.buttons) {
    const dx = event.offsetX - drag.x, dy = event.offsetY - drag.y;
    if (Math.hypot(dx, dy) > 4) drag.moved = true;
    if (drag.moved) {
      panX = drag.panX + dx;
      panY = drag.panY + dy;
      wrap.classList.add('dragging');
      $('hover').hidden = true;
      return;
    }
  }
  hoverAnimal = nearestAnimal(event.offsetX, event.offsetY);
  const label = $('hover');
  label.hidden = !hoverAnimal;
  if (hoverAnimal) {
    label.textContent = `#${hoverAnimal[0]}  ·  gen ${hoverAnimal[7]}  ·  energy ${hoverAnimal[4].toFixed(1)}`;
    label.style.left = `${Math.min(wrap.clientWidth - 175, event.offsetX + 14)}px`;
    label.style.top = `${Math.max(8, event.offsetY - 32)}px`;
  }
});
canvas.addEventListener('pointerup', event => {
  if (drag && !drag.moved) {
    const animal = nearestAnimal(event.offsetX, event.offsetY);
    if (animal) { selectedId = animal[0]; lastKnown = animal; updateInspector(); }
  }
  drag = null;
  wrap.classList.remove('dragging');
});
canvas.addEventListener('pointerleave', () => { $('hover').hidden = true; });
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  const old = transform();
  const worldX = (event.offsetX - old.x) / old.scale;
  const worldY = (event.offsetY - old.y) / old.scale;
  zoom = Math.max(0.75, Math.min(12, zoom * (event.deltaY < 0 ? 1.18 : 1 / 1.18)));
  const next = transform();
  panX += event.offsetX - (next.x + worldX * next.scale);
  panY += event.offsetY - (next.y + worldY * next.scale);
}, {passive: false});
function zoomButton(factor) { zoom = Math.max(.75, Math.min(12, zoom * factor)); }
$('zoom-in').onclick = () => zoomButton(1.4);
$('zoom-out').onclick = () => zoomButton(1 / 1.4);
$('reset').onclick = () => { zoom = 1; panX = 0; panY = 0; };
async function control(path) {
  try { await fetch(path, {method: 'POST', headers: {'X-Live-View': '1'}}); }
  catch { /* Status indicator will show the lost connection. */ }
}
$('pause').onclick = () => control('/api/pause');
$('end').onclick = () => control('/api/stop');
for (const speed of ['1', '4', '0'])
  $(`speed${speed}`).onclick = () => control(`/api/speed/${speed}`);
document.addEventListener('keydown', event => {
  if (event.code === 'Space' && !event.repeat && event.target.tagName !== 'BUTTON') {
    event.preventDefault();
    control('/api/pause');
  }
});
requestAnimationFrame(drawWorld);
poll();
