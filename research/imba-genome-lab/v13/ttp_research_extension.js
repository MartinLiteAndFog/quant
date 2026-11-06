/* v13 research extension: matched REAL legacy TTP matrices and proposed Kraken TTP settings.
 * Never infer uncomputed 0%-activation results from legacy 2%/3%-activation daily factors.
 */
const TTP_RESEARCH_DEFAULTS={activationPct:0,trailPct:1.5,closeFractionPct:100,action:'flip'};
state.krakenTtp={...TTP_RESEARCH_DEFAULTS};
function matchedExitRaster(act,trail){
 const map=new Map();
 META.forEach((m,i)=>{
  if(m.priceMode!=='renko_usd'||m.family!=='ttp')return;
  if(Math.abs(Number(m.act)-act)>1e-5||Math.abs(Number(m.trail)-trail)>1e-5)return;
  map.set(String(m.s)+'|'+Number(m.b).toFixed(5),i);
 });
 const pure=[],ttp=[];
 for(const i of ORIGINAL){
  const m=META[i];
  if(m.family!=='none'||m.priceMode!=='renko_usd')continue;
  const j=map.get(String(m.s)+'|'+Number(m.b).toFixed(5));
  if(j!==undefined){pure.push(i);ttp.push(j);}
 }
 return {pure,ttp,mixed:[...pure,...ttp]};
}
const HIST_EXIT_315=matchedExitRaster(3,1.5);
const HIST_EXIT_21=matchedExitRaster(2,1);
const RESEARCH_POOL_VARIANTS={
 pure315:HIST_EXIT_315.pure,ttp315:HIST_EXIT_315.ttp,mixed315:HIST_EXIT_315.mixed,
 pure21:HIST_EXIT_21.pure,ttp21:HIST_EXIT_21.ttp,mixed21:HIST_EXIT_21.mixed
};
const PREVIOUS_GET_IDS=getPresetIds;
getPresetIds=function(){return RESEARCH_POOL_VARIANTS[state.mode]||PREVIOUS_GET_IDS();};
const PREVIOUS_SERIALIZE=serialize;
serialize=function(){return {...PREVIOUS_SERIALIZE(),krakenTtp:{...state.krakenTtp}};};
function normalizeResearchTtp(x){
 const v=x||{};
 const trail=Number(v.trailPct);
 return {
  activationPct:[0,.5,1,2,3].includes(Number(v.activationPct))?Number(v.activationPct):0,
  trailPct:Number.isFinite(trail)?Math.max(.5,Math.min(4,Math.round(trail*2)/2)):1.5,
  closeFractionPct:Number(v.closeFractionPct)===50?50:100,
  action:v.action==='close'?'close':'flip'
 };
}
function renderTtpControls(){
 const v=state.krakenTtp;
 if(!$('krakenTtpTrail'))return;
 $('krakenTtpTrail').value=String(v.trailPct);$('krakenTtpTrailValue').textContent=number(v.trailPct,1)+' %';
 $('krakenTtpActivation').value=String(v.activationPct);
 $('krakenTtpAction').value=v.action;
 $('krakenTtpFraction').value=String(v.closeFractionPct);
 $('krakenTtpFraction').disabled=v.action==='flip';
 if($('historicalPairsInfo'))$('historicalPairsInfo').textContent=
  'Tatsächlich berechnet: '+HIST_EXIT_315.pure.length+' paarige IMBA + '+HIST_EXIT_315.ttp.length+
  ' TTP 3/1,5 sowie '+HIST_EXIT_21.pure.length+' IMBA + '+HIST_EXIT_21.ttp.length+
  ' TTP 2/1. Nicht das neue Kraken-TTP 0/1,5.';
}
const PREVIOUS_SET_CONTROLS=setControls;
setControls=function(){PREVIOUS_SET_CONTROLS();if($('researchPairPool')&&RESEARCH_POOL_VARIANTS[state.mode])$('researchPairPool').value=state.mode;renderTtpControls();};
function historicalCompare(){
 const out=$('historicalCompareOutput'),chart=$('historicalCompareChart');
 if(state.priceMode!=='renko_usd'){out.textContent='Nur feste USD-Renko-Boxen können mit dem 462er-Raster verglichen werden.';return;}
 const test=[
  ['IMBA pur · paarig',HIST_EXIT_315.pure,null,false],
  ['TTP Alt 3%/1,5%',HIST_EXIT_315.ttp,null,false],
  ['IMBA + TTP Alt',HIST_EXIT_315.mixed,null,false],
  ['Aktueller Pool · ρ≤0,85',state.filtered,.85,false],
  ['Aktueller Pool · ρ≤0,90',state.filtered,.9,false]
 ];
 const traces=[],labels=[];
 for(const [name,base,limit,fallback] of test){
  const pool=base.filter(i=>state.box==='all'||Math.abs(Number(META[i].b)-Number(state.box))<1e-5);
  if(!pool.length){labels.push(name+': keine simulierten Daten');continue;}
  const rows=simulatePortfolio({pool,rank:state.rank,topN:state.topN,guard:state.guard,stopScope:state.stopScope,
    corrLimit:limit===null?state.corrLimit:limit,corrFallback:fallback});
  let peak=1000,dd=0;
  for(const r of rows){peak=Math.max(peak,r.wealth);dd=Math.min(dd,100*(r.wealth/peak-1));}
  labels.push(name+': '+euro(rows[rows.length-1].wealth)+' · DD '+number(dd,1)+' % · n='+pool.length);
  traces.push({type:'scatter',mode:'lines',name,x:rows.map(r=>r.date),y:rows.map(r=>r.wealth)});
 }
 out.textContent=labels.join(' | ')+' · '+state.dataset+' · Top '+state.topN+
  ' · '+state.rank+' Tage Ranking · Stopp '+state.guard+' % · '+state.stopScope+
  '. Vergleich mit gleichen Portfolioregeln, historische Alt-TTP-Signale. Keine Kraken-Flip-Resultate.';
 if(traces.length&&typeof Plotly!=='undefined'){
  chart.style.height='390px';
  Plotly.react(chart,traces,{paper_bgcolor:'#0c2336',plot_bgcolor:'#102b40',
   font:{color:'#c9deed',size:11},margin:{l:60,r:20,t:50,b:55},
   xaxis:{gridcolor:'#284e62'},yaxis:{title:'Depotwert (€)',gridcolor:'#284e62'},
   legend:{orientation:'h',x:0,y:1.18},hovermode:'x unified'},
   {responsive:true,displaylogo:false});
 }
}
async function compareTtpYears(){
 const btn=$('historicalAllYearsBtn'),out=$('historicalAllYearsOutput'),savedYear=state.dataset,savedDay=state.day;
 const savedSelected=state.selected;
 btn.disabled=true;out.textContent='Jahresvergleich laden …';
 const series=[
  ['IMBA',HIST_EXIT_315.pure,null,false],
  ['TTP Alt',HIST_EXIT_315.ttp,null,false],
  ['Mix',HIST_EXIT_315.mixed,null,false],
  ['Korr. 0,85 strikt',null,.85,false],
  ['Korr. 0,90 auffüllen',null,.9,true]
 ];
 const rows=[],failures=[];
 try{
  for(const year of ['2022','2023','2024','2025','2026']){
   out.textContent='Prüfe '+year+' … ('+(rows.length+1)+'/5)';
   await loadYear(year);
   const currentPool=eligible();
   const vals=series.map(([name,base,cap,fallback])=>{
    const pool=(base||currentPool).filter(i=>base===null||
      state.box==='all'||Math.abs(Number(META[i].b)-Number(state.box))<1e-5);
    if(!pool.length)return null;
    const x=simulatePortfolio({pool,rank:state.rank,topN:state.topN,guard:state.guard,
     stopScope:state.stopScope,corrLimit:cap===null?state.corrLimit:cap,
     corrFallback:cap===null?state.corrFallback:fallback},null,true);
    return {value:x.endWealth,dd:x.maxDrawdownPct,count:pool.length};
   });
   rows.push([year,...vals]);
   await new Promise(resolve=>setTimeout(resolve,0));
  }
 }catch(e){failures.push(e.message);}
 finally{
  try{await loadYear(savedYear);state.day=Math.min(savedDay,DAYS-1);state.selected=savedSelected;changePool();}
  catch(e){failures.push('Wiederherstellen: '+e.message);}
  btn.disabled=false;
 }
 const table=document.createElement('table');table.style.cssText='width:100%;font-size:11px;border-collapse:collapse';
 const title=document.createElement('tr');
 for(const h of ['Jahr',...series.map(s=>s[0])]){
  const cell=document.createElement('th');cell.textContent=h;cell.style.padding='5px';title.appendChild(cell);
 }
 table.appendChild(title);
 for(const row of rows){
  const tr=document.createElement('tr');
  for(const value of row){
   const td=document.createElement('td');td.style.cssText='padding:6px;border-top:1px solid #385769';
   td.textContent=typeof value==='string'?value:value?euro(value.value)+' / '+number(value.dd,1)+' % DD':'–';
   tr.appendChild(td);
  }
  table.appendChild(tr);
 }
 out.replaceChildren(document.createTextNode(
  'Einzeln gestartete Jahresdepots zu je 1.000 €, kein Jahresübertrag; aktuelle Top-N/Ranking/Stopp-Regeln. '+
  'TTP Alt 3 % / 1,5 % ist nicht der neue Kraken-Flip. '+(failures.length?'Fehler: '+failures.join('; '):'')
 ),table);
}

function installTTPResearch(){
 const base=$('tradeSweepControls');
 if(!base)throw Error('Top-N-Trade-Sweep UI fehlt');
 const panel=document.createElement('section');panel.className='tradeSweepControls';panel.id='ttpResearchControls';
 panel.innerHTML='<h3>462er-Raster · IMBA gegen TTP / Korrelation</h3>'+
  '<p class="muted">Gleiche Sensitivity-/Box-Koordinaten. Nur bereits vorhandene Alt-TTP-Ergebnisse (Aktivierung 3 % oder 2 %) sind in den Tageskurven enthalten.</p>'+
  '<label for="researchPairPool">Historisch vorhandener Pool</label>'+
  '<select id="researchPairPool"><option value="pure315">462 IMBA pur · paarig</option>'+
  '<option value="ttp315">462 TTP Alt · Aktivierung 3 % / Trail 1,5 %</option>'+
  '<option value="mixed315">Bis 924 IMBA + TTP Alt 3/1,5</option>'+
  '<option value="pure21">462 IMBA pur · paarig mit TTP 2/1</option>'+
  '<option value="ttp21">462 TTP Alt · Aktivierung 2 % / Trail 1 %</option>'+
  '<option value="mixed21">Bis 924 IMBA + TTP Alt 2/1</option></select>'+
  '<p id="historicalPairsInfo" class="muted"></p>'+
  '<div class="tradeSweepButtons"><button id="researchCorr85" type="button">ρ≤0,85 strikt</button>'+
  '<button id="researchCorr90" type="button">ρ≤0,90 auffüllen</button>'+
  '<button id="researchCorrOff" type="button">Korrelation aus</button></div>'+
  '<button type="button" id="historicalCompareBtn">Historische Pools vergleichen</button>'+
  '<button type="button" id="historicalAllYearsBtn">Vergleich 2022–2026</button>'+
  '<div id="historicalAllYearsOutput" class="muted"></div>'+
  '<p id="historicalCompareOutput" class="muted" aria-live="polite"></p><div id="historicalCompareChart"></div>'+
  '<details><summary>Neue Kraken-TTP-Einstellungen (Research-Preset, Backtest ausstehend)</summary>'+
  '<p class="muted">Keine fiktiven Erträge: Änderungen hier werden gespeichert, aber nicht auf die vorhandenen Alt-TTP-Tagesrenditen angewendet.</p>'+
  '<label for="krakenTtpActivation">Aktivierung</label><select id="krakenTtpActivation">'+
  '<option value="0">0 % sofort</option><option value="0.5">0,5 %</option>'+
  '<option value="1">1 %</option><option value="2">2 %</option><option value="3">3 %</option></select>'+
  '<label for="krakenTtpTrail">Trailing-Abstand: <strong id="krakenTtpTrailValue">1,5 %</strong></label>'+
  '<input id="krakenTtpTrail" type="range" min="0.5" max="4" step="0.5" value="1.5">'+
  '<label for="krakenTtpAction">Aktion beim TTP-Stopp</label>'+
  '<select id="krakenTtpAction"><option value="flip">Kraken-Flip → WAIT</option>'+
  '<option value="close">Schließen / Teilgewinn</option></select>'+
  '<label for="krakenTtpFraction">Geschlossener Anteil (nur Schließen)</label>'+
  '<select id="krakenTtpFraction"><option value="100">100 %</option><option value="50">50 %</option></select>'+
  '<p class="muted">Kraken-Flip schließt die alte Position vollständig und eröffnet die Gegenrichtung. Der neue Trade ist zunächst WAIT (SL), bis IMBA erneut TTP aktiviert. Für einen veränderbaren TTP braucht es neue Intraday-Trades und neue Tageskurven.</p>'+
  '</details>';
 base.insertAdjacentElement('afterend',panel);
 const pool=$('researchPairPool');
 const poolDropdown=$('poolMode');
 const poolLabels={
  pure315:'IMBA pur · TTP 3/1,5 gepaart',ttp315:'TTP Alt 3% / 1,5% · gepaart',
  mixed315:'IMBA + TTP Alt 3/1,5',pure21:'IMBA pur · TTP 2/1 gepaart',
  ttp21:'TTP Alt 2% / 1%',mixed21:'IMBA + TTP Alt 2/1'
 };
 for(const [value,label] of Object.entries(poolLabels)){
  if(![...poolDropdown.options].some(o=>o.value===value)){
   const o=document.createElement('option');o.value=value;o.textContent=label;poolDropdown.appendChild(o);
  }
 }
 const nativePoolChange=poolDropdown.onchange;
 poolDropdown.onchange=async e=>{
  if(RESEARCH_POOL_VARIANTS[e.target.value]){
   const mode=e.target.value;const fam=mode.startsWith('pure')?'none':mode.startsWith('ttp')?'ttp':'mix';
   state.priceMode='renko_usd';state.box='all';
   state.familyEnabled={none:fam!=='ttp',tp:false,ttp:fam!=='none',both:false};
   state.family=familySelectionMode();
  }
  await nativePoolChange(e);
 };
 pool.onchange=e=>{
  state.mode=e.target.value;state.priceMode='renko_usd';state.box='all';
  const fam=e.target.value.startsWith('pure')?'none':e.target.value.startsWith('ttp')?'ttp':'mix';
  state.familyEnabled={none:fam!=='ttp',tp:false,ttp:fam!=='none',both:false};
  state.family=familySelectionMode();
  if($('poolMode')&&![...$('poolMode').options].some(o=>o.value===state.mode)){
   const opt=document.createElement('option');opt.value=state.mode;opt.textContent=e.target.selectedOptions[0].textContent;$('poolMode').appendChild(opt);
  }
  syncModeControls();setControls();changePool();
 };
 $('researchCorr85').onclick=()=>{state.corrLimit=.85;state.corrFallback=false;setControls();changePool();};
 $('researchCorr90').onclick=()=>{state.corrLimit=.9;state.corrFallback=true;setControls();changePool();};
 $('researchCorrOff').onclick=()=>{state.corrLimit=null;state.corrFallback=false;setControls();changePool();};
 $('historicalCompareBtn').onclick=historicalCompare;
 $('historicalAllYearsBtn').onclick=compareTtpYears;
 $('krakenTtpActivation').onchange=e=>{state.krakenTtp.activationPct=Number(e.target.value);};
 $('krakenTtpTrail').oninput=e=>{state.krakenTtp.trailPct=Number(e.target.value);renderTtpControls();};
 $('krakenTtpAction').onchange=e=>{state.krakenTtp.action=e.target.value==='close'?'close':'flip';renderTtpControls();};
 $('krakenTtpFraction').onchange=e=>{state.krakenTtp.closeFractionPct=Number(e.target.value)===50?50:100;};
 const oldLoad=$('loadPreset').onchange;
 $('loadPreset').onchange=async e=>{
  const file=e.target.files[0];
  if(file){try{const x=JSON.parse(await file.text());state.krakenTtp=normalizeResearchTtp(x.krakenTtp);}catch(err){console.error(err);}}
  await oldLoad(e);renderTtpControls();
 };
 const select=$('correlationLimit');
 for(const val of [.5,.65,.7,.75,.8,.95]){
  if(![...select.options].some(o=>o.value===String(val))){
   const opt=document.createElement('option');opt.value=String(val);opt.textContent='Maximal +'+val.toFixed(2);select.appendChild(opt);
  }
 }
 if(state.corrLimit!==null)select.value=String(state.corrLimit);
 renderTtpControls();
}
