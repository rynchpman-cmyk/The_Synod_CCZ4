"use strict";

const COLORS = { blue: "#46a8ff", orange: "#ff9d3f", green: "#61df9b", amber: "#ffc45b", cyan: "#66e4ff", violet: "#c58cff", grid: "#263651", text: "#7e8ba3" };
const state = { catalog: null, data: null, scenario: null, frame: 0, playing: false, speed: 1, field: "a_z", timer: null, heatmaps: {} };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const format = (value, digits = 3) => Math.abs(value) >= 1e-2 && Math.abs(value) < 1e4 ? value.toFixed(digits) : value.toExponential(2);

async function fetchJSON(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

function scenarioId() {
  const requested = location.hash.replace("#", "");
  return state.catalog.scenarios.some(item => item.id === requested && item.available) ? requested : "interaction_dominant";
}

function setAccent(color) { document.documentElement.style.setProperty("--accent", color); }

function buildNavigation() {
  const nav = $("#scenario-nav");
  nav.innerHTML = state.catalog.scenarios.filter(s => s.available).map(s => `<button data-scenario="${s.id}">${s.title}</button>`).join("");
  nav.addEventListener("click", event => {
    const button = event.target.closest("button[data-scenario]");
    if (button) location.hash = button.dataset.scenario;
  });
}

function interpretation(id) {
  if (id === "interaction_dominant") return `
    <p>A slowly relaxing target overlaps a heavy, strongly coupled repair mode. Watch for:</p>
    <ul><li><strong>Exchange takeover:</strong> interaction energy briefly exceeds bare Proca energy near t ≈ 1.45.</li>
    <li><strong>Shell train:</strong> alternating-sign mismatch bands leave the two matter lobes.</li>
    <li><strong>Five reversals:</strong> the initial-profile projection repeatedly changes sign while the wake moves outward.</li></ul>`;
  return `<p>A rapid target quench leaves the dynamical field behind as an underdamped wake. Watch for:</p>
    <ul><li><strong>Immediate mismatch:</strong> A<sup>★</sup> collapses before A can follow.</li>
    <li><strong>Clean ringing:</strong> five phase reversals appear as alternating spacetime bands.</li>
    <li><strong>Localized delay:</strong> strong matter coupling slows part of the pulse while the rest radiates outward.</li></ul>`;
}

async function loadScenario(id) {
  stopPlayback();
  $("#loading").classList.remove("hidden");
  try {
    state.scenario = id;
    state.data = await fetchJSON(`/api/showcase/${id}`);
    state.frame = 0; state.heatmaps = {};
    setAccent(state.data.accent);
    $$("#scenario-nav button").forEach(button => button.classList.toggle("active", button.dataset.scenario === id));
    $("#scenario-code").textContent = id.toUpperCase();
    $("#scenario-title").textContent = state.data.title;
    $("#scenario-subtitle").textContent = state.data.subtitle;
    const c = state.data.summary.case;
    $("#parameter-strip").innerHTML = [["m_A", c.mass], ["τ★", c.tau_star], ["COUPLING", c.coupling], ["ρ PEAK", c.peak_density], ["β TARGET", c.beta_target]].map(([label,value]) => `<div class="parameter"><span>${label}</span><strong>${value}</strong></div>`).join("");
    $("#interpretation").innerHTML = interpretation(id);
    $("#gauss-value").textContent = format(state.data.summary.final_gauss_l2, 2);
    $("#time-slider").max = state.data.arrays.time.length - 1;
    $("#time-slider").value = 0;
    createHeatmaps(); renderAll();
  } catch (error) { showError(error.message); }
  finally { $("#loading").classList.add("hidden"); }
}

function valueRange(matrix, symmetric = true) {
  let min = Infinity, max = -Infinity;
  matrix.forEach(row => row.forEach(value => { min = Math.min(min, value); max = Math.max(max, value); }));
  if (symmetric) { const extent = Math.max(Math.abs(min), Math.abs(max), 1e-30); return [-extent, extent]; }
  return [min, max];
}

function colorDiverging(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[31,72,163],[85,140,211],[225,230,235],[239,138,98],[170,36,45]];
  const scaled = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(scaled)), f = scaled - i;
  return stops[i].map((v,k) => Math.round(v + f * (stops[i+1][k]-v)));
}
function colorEnergy(t) {
  t = Math.pow(Math.max(0, Math.min(1, t)), .42);
  const stops = [[5,9,20],[26,38,81],[47,118,142],[242,175,70],[255,242,170]];
  const scaled=t*(stops.length-1), i=Math.min(stops.length-2,Math.floor(scaled)), f=scaled-i;
  return stops[i].map((v,k)=>Math.round(v+f*(stops[i+1][k]-v)));
}

function createHeatmap(matrix, symmetric) {
  const rows=matrix.length, cols=matrix[0].length, canvas=document.createElement("canvas"); canvas.width=cols; canvas.height=rows;
  const context=canvas.getContext("2d"), image=context.createImageData(cols,rows), range=valueRange(matrix,symmetric);
  for(let y=0;y<rows;y++) for(let x=0;x<cols;x++) {
    const value=matrix[y][x], t=(value-range[0])/Math.max(range[1]-range[0],1e-30), color=symmetric?colorDiverging(t):colorEnergy(t), offset=4*(y*cols+x);
    image.data[offset]=color[0]; image.data[offset+1]=color[1]; image.data[offset+2]=color[2]; image.data[offset+3]=255;
  }
  context.putImageData(image,0,0); return {canvas, range};
}

function createHeatmaps() {
  const a=state.data.arrays;
  const mismatch=a.a_z.map((row,i)=>row.map((v,j)=>v-a.astar_z[i][j]));
  state.heatmaps = { a_z:createHeatmap(a.a_z,true), mismatch:createHeatmap(mismatch,true), energy:createHeatmap(a.energy,false), phi:createHeatmap(a.phi,true) };
}

function setupCanvas(canvas) {
  const rect=canvas.getBoundingClientRect(), ratio=window.devicePixelRatio||1;
  const width=Math.max(1,Math.round(rect.width*ratio)), height=Math.max(1,Math.round(rect.height*ratio));
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  const ctx=canvas.getContext("2d"); ctx.setTransform(ratio,0,0,ratio,0,0); return {ctx,width:rect.width,height:rect.height};
}

function drawAxes(ctx,w,h,pad,xMin,xMax,yMin,yMax,xLabel="") {
  ctx.strokeStyle=COLORS.grid;ctx.lineWidth=1;ctx.fillStyle=COLORS.text;ctx.font="10px system-ui";ctx.textAlign="center";
  for(let i=0;i<=4;i++){const x=pad.l+(w-pad.l-pad.r)*i/4;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,h-pad.b);ctx.stroke();ctx.fillText((xMin+(xMax-xMin)*i/4).toFixed(1),x,h-pad.b+16);}
  ctx.textAlign="right";for(let i=0;i<=4;i++){const y=pad.t+(h-pad.t-pad.b)*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillText((yMax-(yMax-yMin)*i/4).toFixed(2),pad.l-7,y+3);}
  if(xLabel){ctx.textAlign="center";ctx.fillText(xLabel,(pad.l+w-pad.r)/2,h-3);}
}

function drawLineChart(canvas, x, series, options={}) {
  const {ctx,width:w,height:h}=setupCanvas(canvas), pad={l:48,r:14,t:12,b:30};ctx.clearRect(0,0,w,h);
  let yMin=options.yMin??Math.min(...series.flatMap(s=>s.values)), yMax=options.yMax??Math.max(...series.flatMap(s=>s.values));
  if(options.includeZero!==false){yMin=Math.min(0,yMin);yMax=Math.max(0,yMax);} const span=Math.max(yMax-yMin,1e-12);yMin-=.06*span;yMax+=.06*span;
  const xMin=x[0],xMax=x[x.length-1],px=v=>pad.l+(v-xMin)/(xMax-xMin)*(w-pad.l-pad.r),py=v=>pad.t+(yMax-v)/(yMax-yMin)*(h-pad.t-pad.b);
  drawAxes(ctx,w,h,pad,xMin,xMax,yMin,yMax,options.xLabel||"");
  series.forEach(s=>{ctx.strokeStyle=s.color;ctx.lineWidth=s.width||2;ctx.beginPath();s.values.forEach((v,i)=>i?ctx.lineTo(px(x[i]),py(v)):ctx.moveTo(px(x[i]),py(v)));ctx.stroke();});
  if(options.cursor!==undefined){const cx=px(options.cursor);ctx.strokeStyle="rgba(255,255,255,.65)";ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(cx,pad.t);ctx.lineTo(cx,h-pad.b);ctx.stroke();ctx.setLineDash([]);}
}

function renderHeatmap() {
  const canvas=$("#heatmap"), {ctx,width:w,height:h}=setupCanvas(canvas), source=state.heatmaps[state.field], pad={l:48,r:18,t:12,b:30};ctx.clearRect(0,0,w,h);ctx.imageSmoothingEnabled=true;
  ctx.drawImage(source.canvas,pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);
  ctx.strokeStyle=COLORS.grid;ctx.strokeRect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);
  const a=state.data.arrays, t=a.time[state.frame], y=pad.t+t/a.time.at(-1)*(h-pad.t-pad.b);ctx.strokeStyle="#fff";ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
  ctx.fillStyle=COLORS.text;ctx.font="10px system-ui";ctx.textAlign="center";for(let i=0;i<=4;i++)ctx.fillText((a.x[0]+(a.x.at(-1)-a.x[0])*i/4).toFixed(1),pad.l+(w-pad.l-pad.r)*i/4,h-8);
  ctx.save();ctx.translate(12,(pad.t+h-pad.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText("SIMULATION TIME",0,0);ctx.restore();
  $("#heatmap-cursor-label").textContent=`t = ${t.toFixed(2)}  ·  range [${format(source.range[0],2)}, ${format(source.range[1],2)}]`;
}

function renderMetrics() {
  const a=state.data.arrays,i=state.frame,initial=a.a_l2[0],ratio=a.interaction_energy[i]/Math.max(a.proca_energy[i],1e-30);
  const items=[
    ["MISMATCH / A₀",a.mismatch_l2[i]/initial,Math.min(100,100*a.mismatch_l2[i]/initial),COLORS.green],
    ["INTERACTION / PROCA",ratio,Math.min(100,70*ratio),COLORS.amber],
    ["EXTERIOR ENERGY",a.exterior_energy_fraction[i],100*a.exterior_energy_fraction[i],COLORS.cyan],
    ["PROFILE PHASE",a.profile_projection[i],50+45*a.profile_projection[i],COLORS.violet],
    ["SPECTRAL ENTROPY",a.spectral_entropy[i],100*a.spectral_entropy[i],COLORS.blue],
  ];
  $("#metrics").innerHTML=items.map(([label,value,fill,color],n)=>`<div class="metric" style="--fill:${Math.max(0,fill)}%;--metric-color:${color}"><span>${label}</span><strong>${format(value)}</strong><em>0${n+1}</em></div>`).join("");
}

function renderCharts() {
  const a=state.data.arrays,i=state.frame,t=a.time[i],initial=a.a_l2[0],mismatch=a.a_z[i].map((v,j)=>v-a.astar_z[i][j]);
  drawLineChart($("#profile-chart"),a.x,[{values:a.a_z[i],color:COLORS.blue},{values:a.astar_z[i],color:COLORS.orange},{values:mismatch,color:COLORS.green}],{xLabel:"position x"});
  drawLineChart($("#amplitude-chart"),a.time,[{values:a.a_l2.map(v=>v/initial),color:COLORS.blue},{values:a.astar_l2.map(v=>v/a.astar_l2[0]),color:COLORS.orange},{values:a.mismatch_l2.map(v=>v/initial),color:COLORS.green}],{cursor:t,xLabel:"time"});
  drawLineChart($("#exchange-chart"),a.time,[{values:a.interaction_energy.map((v,j)=>v/Math.max(a.proca_energy[j],1e-30)),color:COLORS.amber},{values:a.exterior_energy_fraction,color:COLORS.cyan},{values:a.profile_projection,color:COLORS.violet}],{cursor:t,xLabel:"time"});
  drawLineChart($("#energy-chart"),a.x,[{values:a.energy[i],color:COLORS.amber,width:2.4}],{xLabel:"position x",yMin:0});
  const peakIndex=a.energy[i].reduce((best,v,j)=>v>a.energy[i][best]?j:best,0);$("#peak-location").textContent=`Peak |x| = ${Math.abs(a.x[peakIndex]).toFixed(2)}`;
}

function renderAll() {
  if(!state.data)return; const t=state.data.arrays.time[state.frame];
  $("#time-slider").value=state.frame;$("#time-readout").textContent=`t = ${t.toFixed(3)}`;$("#profile-time").textContent=`t = ${t.toFixed(2)}`;
  renderMetrics();renderHeatmap();renderCharts();
}

function setFrame(frame){state.frame=Math.max(0,Math.min(state.data.arrays.time.length-1,frame));renderAll();}
function startPlayback(){if(state.playing)return;state.playing=true;$("#play-button span").textContent="Ⅱ";state.timer=setInterval(()=>{if(state.frame>=state.data.arrays.time.length-1)setFrame(0);else setFrame(state.frame+1);},50/state.speed);}
function stopPlayback(){state.playing=false;clearInterval(state.timer);state.timer=null;const icon=$("#play-button span");if(icon)icon.textContent="▶";}
function togglePlayback(){state.playing?stopPlayback():startPlayback();}
function showError(message){const banner=$("#error-banner");banner.textContent=message;banner.hidden=false;setTimeout(()=>banner.hidden=true,8000);}

function bindEvents(){
  window.addEventListener("hashchange",()=>loadScenario(scenarioId()));window.addEventListener("resize",()=>requestAnimationFrame(renderAll));
  $("#play-button").addEventListener("click",togglePlayback);$("#time-slider").addEventListener("input",event=>{stopPlayback();setFrame(+event.target.value);});
  $("#speed-select").addEventListener("change",event=>{state.speed=+event.target.value;if(state.playing){stopPlayback();startPlayback();}});
  $("#field-tabs").addEventListener("click",event=>{const button=event.target.closest("button[data-field]");if(!button)return;state.field=button.dataset.field;$$('#field-tabs button').forEach(b=>b.classList.toggle('active',b===button));const names={a_z:"Dynamical field A_z",mismatch:"Active mismatch A_z − A_z★",energy:"Repair-sector energy density",phi:"Constrained normal field Φ"};$("#heatmap-title").textContent=names[state.field];renderHeatmap();});
  $("#heatmap").addEventListener("click",event=>{const rect=event.target.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(event.clientY-rect.top-12)/(rect.height-42)));stopPlayback();setFrame(Math.round(fraction*(state.data.arrays.time.length-1)));});
  document.addEventListener("keydown",event=>{if(event.code==="Space"){event.preventDefault();togglePlayback();}if(event.code==="ArrowRight")setFrame(state.frame+1);if(event.code==="ArrowLeft")setFrame(state.frame-1);});
}

async function init(){try{state.catalog=await fetchJSON("/api/catalog");buildNavigation();bindEvents();if(!location.hash)location.hash="#interaction_dominant";else await loadScenario(scenarioId());}catch(error){showError(error.message);$("#loading").classList.add("hidden");}}
init();
