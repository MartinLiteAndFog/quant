/* Quant Genome v8 — every candidate belongs to an individually simulated daily curve.
 * Analytical 1x charts, research 10x treasury, no TradingView execution claims.
 */
const $=id=>document.getElementById(id);
const N=META.length, DAYS=DATES.length, W=90+DAYS;
const FAMILY_COLORS={none:'#66a9f1',tp:'#e8ac66',ttp:'#5ad5ab',both:'#bd91ed'};
const GENE_ORDER=[['imba','⌁','IMBA','#48cbd5'],['renko','▦','RENKO','#70a8ed'],['candle','▤','CANDLES','#78d3ef'],['tp','↗','TP','#ecb56e'],['ttp','↝','TTP','#b995ef']];
const EXIT_NAMES=['Ohne TP/TTP','TP 6 %','TP 4 %','TP 3 %','TP6 + TTP3/1,5','TP 2 %','TP4 + TTP2/1','TTP 3/1,5','TTP 2/1'];
const AXES={s:'Sensitivity',b:'Renko-Box (USD)',tp:'TP-Ziel (%)',act:'TTP-Aktivierung (%)',trail:'TTP-Abstand (%)',exit:'Exit-Programm'};
const INTERVALS=[14,30,90,0];
let state={priceMode:'renko_usd',day:DAYS-1,horizon:30,x:'tp',y:'s',mode:'all4158',family:'tp',box:'all',dataset:'2026',
  aggregate:'max',chart:'3d',points:'top6',guard:30,rank:14,manual:new Set(),selected:-1,
  playing:null,dir:1,port:null,portKey:'',filtered:[],gridN:0,cacheKey:'',grid:null,rendering:false,pending:false};
const IDMAP=new Map(META.map((m,i)=>[m.id,i]));
function decodeF32(b64){const raw=atob(b64), bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Float32Array(bytes.buffer);}
const F=decodeF32(F_B64), P22=decodeF32(P22_B64);
const PREFIX=new Float64Array(N*(W+1));
for(let i=0;i<N;i++){
  const src=i*W,dst=i*(W+1);
  for(let j=0;j<W;j++) PREFIX[dst+j+1]=PREFIX[dst+j]+Math.log(Math.max(F[src+j],1e-15));
}
function dayFactor(i,d){return Number(F[i*W+90+d]);}
function rPct(i,end,window){ // end = index of last included day relative to 2026 Jan1; -1 is 2025-12-31
  const p=90+end+1,left=window?Math.max(0,p-window):90;
  if(p<=left)return 0;
  const start=i*(W+1);
  return 100*Math.expm1(PREFIX[start+p]-PREFIX[start+left]);
}
function value(i){return state.dataset==='2022'?(META[i].has2022?P22[i]:NaN):rPct(i,state.day,state.horizon);}
function score(i,day){return rPct(i,day-1,state.rank);}
function number(v,dec=1){return Number(v).toLocaleString('de-DE',{minimumFractionDigits:dec,maximumFractionDigits:dec});}
function euro(v){return Number(v).toLocaleString('de-DE',{style:'currency',currency:'EUR',maximumFractionDigits:2});}
function pct(v){return (v>0?'+':'')+number(v,1)+' %';}
function geneValue(i,k){const m=META[i];return m[k];}
function realAxisValue(i,k){const m=META[i];if(k==='tp')return m.tpValue;if(k==='act')return m.act;if(k==='trail')return m.trail;return Number(m[k]);}
function getPresetIds(){
 if(state.mode==='original462')return ORIGINAL;
 if(state.mode==='top100')return RAW100;
 if(state.mode==='balanced100')return BAL100;
 if(state.mode==='manual')return [...state.manual];
 return ALL;
}
function pickAxisCoords(vals,n){
 const uniq=[...new Set(vals)].sort((a,b)=>a-b);
 if(uniq.length<=n)return uniq;
 const chosen=new Set();for(let j=0;j<n;j++)chosen.add(uniq[Math.round(j*(uniq.length-1)/(n-1))]);
 return [...chosen];
}
function eligible(){
 let ids=getPresetIds().filter(i=>META[i].priceMode===state.priceMode && (state.dataset!=='2022'||META[i].has2022) && (state.family==='all'||META[i].family===state.family));
 if(state.box!=='all' && state.priceMode!=='candle')ids=ids.filter(i=>Math.abs(META[i].b-Number(state.box))<.00001);
 if(state.mode==='grid20'||state.mode==='grid30'){
  const k=state.mode==='grid20'?20:30;
  const xv=new Set(pickAxisCoords(ids.map(i=>realAxisValue(i,state.x)),k));
  const yv=new Set(pickAxisCoords(ids.map(i=>realAxisValue(i,state.y)),k));
  ids=ids.filter(i=>xv.has(realAxisValue(i,state.x))&&yv.has(realAxisValue(i,state.y)));
 }
 return ids;
}
function populateSelector(sel,items,selected){$(sel).innerHTML=items.map(([v,txt])=>`<option value="${v}">${txt}</option>`).join('');$(sel).value=selected;}
function initOptions(){
 populateSelector('axisX',Object.entries(AXES),'tp');populateSelector('axisY',Object.entries(AXES),'s');
 populateSelector('boxFilter',[['all','Alle Boxgrößen (Aggregation)'],...BOXES.map(b=>[b.toFixed(2),b.toFixed(2)+' USD (exakt)'])],'all');
}
function sameCell(a,b){return Number(a)===Number(b);}
function buildHeatmap(){
 const ids=state.filtered;
 const xs=[...new Set(ids.map(i=>realAxisValue(i,state.x)))].sort((a,b)=>a-b);
 const ys=[...new Set(ids.map(i=>realAxisValue(i,state.y)))].sort((a,b)=>a-b);
 const groups=new Map();
 for(const i of ids){const x=realAxisValue(i,state.x),y=realAxisValue(i,state.y);const key=x+'|'+y;
  if(!groups.has(key))groups.set(key,[]);groups.get(key).push(i);}
 const z=[],info=[],custom=[],counts=[],winners=[];
 let occupied=0;
 for(const y of ys){const row=[],ir=[],cr=[],nr=[],wr=[];
  for(const x of xs){const botids=groups.get(x+'|'+y)||[];
   if(!botids.length){row.push(null);ir.push('Nicht getestet');cr.push(null);nr.push(0);wr.push(-1);continue;}
   occupied++;let vals=botids.map(i=>value(i));let best=botids[0],bestValue=-Infinity,sum=0;
   for(let k=0;k<botids.length;k++){sum+=vals[k];if(vals[k]>bestValue){bestValue=vals[k];best=botids[k];}}
   const zval=state.aggregate==='max'?bestValue:sum/vals.length;
   row.push(zval);cr.push(best);nr.push(botids.length);wr.push(best);
   ir.push(`${botids.length} tatsächlich berechnete Strategien<br>Wert (${state.aggregate==='max'?'beste':'Durchschnitt'}): ${pct(zval)}<br>Bester Bot: ${META[best].id}<br>Rendite: ${pct(bestValue)}`);
  }
  z.push(row);info.push(ir);custom.push(cr);counts.push(nr);winners.push(wr);
 }
 return {xs,ys,z,info,custom,counts,winners,groups,occupied};
}
function choosePortfolio(){
 const pool=state.filtered,hash=pool.join(',')+'|'+state.guard+'|'+state.rank;
 if(hash===state.portKey&&state.port)return state.port;
 let reserve=1000,cash=0,basis=0,paid=0,payoutN=0,stopN=0,milestone=0;
 const h=[0,0,0,0,0,0],ids=[-1,-1,-1,-1,-1,-1],stopped=[false,false,false,false,false,false];
 const idx=[1,1,1,1,1,1],hi=[1,1,1,1,1,1];
 const results=[];const sum=a=>a.reduce((x,y)=>x+y,0);
 for(let d=0;d<DAYS;d++){
  const date=DATES[d],newMonth=date.endsWith('-01'),newCycle=d%14===0;
  let sub=0;
  if(newMonth){sub=Math.min(reserve,Math.max(100,.2*reserve));reserve-=sub;cash+=sub;basis+=sub;}
  if(newCycle){cash+=sum(h);h.fill(0);ids.fill(-1);stopped.fill(false);idx.fill(1);hi.fill(1);
   const scored=pool.map(i=>[i,score(i,d)]).filter(v=>v[1]>0).sort((a,b)=>b[1]-a[1]||a[0]-b[0]).slice(0,6);
   scored.forEach((v,k)=>ids[k]=v[0]);const n=scored.length;
   if(n&&cash>0){for(let k=0;k<n;k++)h[k]=cash/n;cash=0;}
  }else if(sub>0){const active=ids.filter((i,k)=>i>=0&&!stopped[k]).length;
   if(active){for(let k=0;k<6;k++)if(ids[k]>=0&&!stopped[k])h[k]+=sub/active;cash-=sub;}
  }
  for(let k=0;k<6;k++)if(ids[k]>=0&&!stopped[k]&&h[k]>1e-10){
   const factor=Math.max(0,1+10*(dayFactor(ids[k],d)-1));h[k]*=factor;idx[k]*=factor;hi[k]=Math.max(hi[k],idx[k]);
  }
  let nav=cash+sum(h),profit=Math.max(0,nav+paid-basis),ratio=basis>0?profit/basis:0;
  if(basis>0&&ratio>=Math.max(1,milestone+1)-1e-10){const take=Math.min(nav*.5,.5*profit);
   if(take>1e-10){const c=Math.min(cash,take);cash-=c;const rest=take-c;
    if(rest>1e-10&&sum(h)>0){const scale=Math.max(0,1-rest/sum(h));for(let k=0;k<6;k++)h[k]*=scale;}
    reserve+=take;paid+=take;payoutN++;milestone++;}
  }
  if(state.guard>0){for(let k=0;k<6;k++)if(ids[k]>=0&&!stopped[k]&&idx[k]/Math.max(hi[k],1e-12)-1<=-state.guard/100){
    stopped[k]=true;cash+=h[k];h[k]=0;stopN++;
  }}
  const trading=cash+sum(h),live=ids.filter((i,k)=>i>=0&&!stopped[k]&&h[k]>1e-10);
  results.push({date,wealth:reserve+trading,reserve,trading,cash,ids:ids.slice(),h:h.slice(),stopped:stopped.slice(),live,payoutN,stopN,score:ids.map(i=>i>=0?score(i,Math.floor(d/14)*14):null)});
 }
 state.portKey=hash;state.port=results;
 return results;
}
function refreshPool(){state.filtered=eligible();state.grid=buildHeatmap();state.portKey='';choosePortfolio();setStatus();renderChart();}
function familyName(f){return ({none:'Nur IMBA',tp:'TP ohne TTP',ttp:'TTP ohne TP',both:'TP + TTP'})[f]||'Alle';}
function setStatus(){const d=choosePortfolio()[state.day];const len=state.filtered.length;const chosen=d.ids.filter(i=>i>=0).length;
 $('candidateCount').textContent=number(len,0);$('cellCount').textContent=number(state.grid.occupied,0);$('activeCount').textContent=d.live.length+'/6';
 $('wealth').textContent=euro(d.wealth);$('wealthChange').textContent=pct(100*(d.wealth/1000-1));
 $('reserve').textContent=euro(d.reserve);$('trading').textContent=euro(d.trading);
 $('payouts').textContent=d.payoutN;$('stops').textContent=d.stopN;
 $('dayLabel').textContent=DATES[state.day];$('slider').value=state.day;
 $('liveNote').textContent=`${chosen} gewählt · ${d.live.length} handeln · ${chosen-d.live.length} gestoppt/ohne Einsatz`;
 $('plotCountNote').textContent=`${len} echte Bot-Kurven · ${state.grid.occupied} belegte Parameterzellen · ${state.priceMode==='candle'?'5-Minuten-Candles':state.priceMode==='renko_pct'?'Prozent-Renko':'USD-Renko'} · ${state.box==='all'?'Alle getesteten Größen': 'Box '+state.box} · 10× nur im Portfolio`;
 $('dataWarning').textContent=state.dataset==='2022'?'2022: Nur ursprünglicher USD-Renko-Sweep verfügbar. Prozent-Renko und Candles wurden nur 2025/2026 neu berechnet. Portfolio bleibt 2026.':'2026: Alle sichtbaren Bot-Kurven wurden für den jeweils gewählten Marktmodus berechnet (5m, Forschungsmodell; keine TradingView-Parität).';
 renderGeneCards(d);renderTop6(d);renderEquityChart();
}
function renderGeneCards(d){const scope=$('geneScope').value;
 const list=scope==='active'?d.live:scope==='selected'?d.ids.filter(i=>i>=0):state.filtered;
 $('genes').innerHTML=GENE_ORDER.map(([key,icon,label,color])=>{
   const count=list.filter(i=>key==='imba'||(key==='renko'&&META[i].priceMode!=='candle')||(key==='candle'&&META[i].priceMode==='candle')||META[i][key]===true).length;
   return `<div class="gene" style="--gene:${color}"><span>${icon} ${label}</span><small>${count}</small></div>`;
 }).join('');
}
function renderTop6(d){const el=$('top6List');
 const items=d.ids.map((i,k)=>i<0?null:{i,k}).filter(Boolean);
 el.innerHTML=items.length?items.map(({i,k})=>{const m=META[i],on=d.live.includes(i),p=d.h[k];
  return `<button class="botrow" data-id="${i}"><span class="statusDot" style="background:${on?'#e7be67':'#67778c'}"></span><span>${m.id}</span><strong>${on?euro(p):'Pause'}</strong></button>`;
 }).join(''):'<div class="muted">Kein Bot hat eine positive Vorperiodenrendite.</div>';
 el.querySelectorAll('button').forEach(x=>x.onclick=()=>inspect(+x.dataset.id));
}
function renderEquityChart(){const p=choosePortfolio(),d=state.day,lo=Math.max(0,d-90),arr=p.slice(lo,d+1);
 const values=arr.map(a=>a.wealth),mi=Math.min(...values),ma=Math.max(...values),scale=Math.max(ma-mi,1);
 const path=values.map((v,j)=>`${arr.length===1?0:j/(arr.length-1)*290},${42-(v-mi)/scale*36}`).join(' ');
 $('spark').innerHTML=`<svg viewBox="0 0 290 45" preserveAspectRatio="none" width="100%" height="45"><polyline points="${path}" fill="none" stroke="#5dc9cc" stroke-width="2"/></svg>`;
}
function inspect(id){if(!Number.isInteger(id)||id<0||id>=N)return;state.selected=id;
 const m=META[id],d=choosePortfolio()[state.day];const isLive=d.live.includes(id),chosen=d.ids.includes(id);
 $('inspectTitle').textContent=m.id;
 $('inspectGenes').innerHTML=`<span>⌁ IMBA S${m.s}</span><span>${m.priceMode==='candle'?'▤ Candles 5m':m.priceMode==='renko_pct'?'▦ Renko '+number(m.b,1)+' %':'▦ Renko '+number(m.b,2)+' USD'}</span>`+
 (m.tp?`<span>↗ TP ${m.tpValue}%</span>`:'')+(m.ttp?`<span>↝ TTP ${m.act}/${m.trail}%</span>`:'');
 $('inspectReturn').textContent=`2026 YTD: ${pct(rPct(id,state.day,0))} · 14T: ${pct(rPct(id,state.day,14))} · 30T: ${pct(rPct(id,state.day,30))} · 90T: ${pct(rPct(id,state.day,90))}`;
 $('inspectStatus').textContent=isLive?'● handelt':chosen?'■ gewählt, pausiert':'○ Kandidat / nicht ausgewählt';
 $('manualAdd').disabled=state.manual.has(id);$('manualRemove').disabled=!state.manual.has(id);
}
function layoutAxes(){const axis=k=>({title:{text:k==='b'?(state.priceMode==='renko_pct'?'Renko-Box (%)':state.priceMode==='candle'?'Candles · ohne Box':'Renko-Box (USD)'):AXES[k],font:{size:12}},gridcolor:'#304b61',zerolinecolor:'#304b61',showbackground:true,backgroundcolor:'#102437'});
 const x=axis(state.x),y=axis(state.y);
 if(state.x==='exit'){x.tickvals=[0,1,2,3,4,5,6,7,8];x.ticktext=['none','TP6','TP4','TP3','TP6+TTP','TP2','TP4+TTP','TTP3','TTP2'];}
 if(state.y==='exit'){y.tickvals=[0,1,2,3,4,5,6,7,8];y.ticktext=['none','TP6','TP4','TP3','TP6+TTP','TP2','TP4+TTP','TTP3','TTP2'];}
 return {x,y};}
function makePlotData(){const g=state.grid,{x,y}=layoutAxes(),is3=state.chart==='3d';
 const traces=[];if(g.xs.length&&g.ys.length){
  const scale=[[0,'#ca596c'],[.38,'#75465f'],[.5,'#23455f'],[.65,'#26796f'],[1,'#65cea7']];
  if(is3 && g.xs.length>=2 && g.ys.length>=2){
   traces.push({type:'surface',x:g.xs,y:g.ys,z:g.z,colorscale:scale,opacity:.90,connectgaps:false,
    customdata:g.custom,hovertext:g.info,hoverinfo:'text',name:'Getestete Varianten',showscale:true,colorbar:{title:'1× %',len:.48,thickness:11}});
  } else if(!is3){traces.push({type:'heatmap',x:g.xs,y:g.ys,z:g.z,colorscale:scale,connectgaps:false,
   customdata:g.custom,hovertext:g.info,hoverinfo:'text',name:'Getestete Varianten',showscale:true,colorbar:{title:'1× %',thickness:11}});}
 }
 const portfolio=choosePortfolio()[state.day],live=new Set(portfolio.live),chosen=new Set(portfolio.ids.filter(i=>i>=0));
 let dotIds=state.points==='all'?state.filtered:state.points==='manual'?[...state.manual]:portfolio.ids.filter(i=>i>=0);
 const dots=dotIds.map(i=>({i,m:META[i],v:value(i),trade:live.has(i),chosen:chosen.has(i)}));
 if(dots.length){const args={x:dots.map(a=>realAxisValue(a.i,state.x)),y:dots.map(a=>realAxisValue(a.i,state.y)),
   customdata:dots.map(a=>a.i),hovertext:dots.map(a=>`${a.m.id}<br>${pct(a.v)} (${state.dataset})<br>${a.trade?'HANDELT':a.chosen?'GESTOPPT':'KANDIDAT'}`),hoverinfo:'text',
   marker:{size:dots.map(a=>a.trade?9:a.chosen?7:Math.min(6,2.5+Math.abs(a.v)/28)),
   color:dots.map(a=>a.trade?'#ffcf67':a.chosen?'#96abc4':a.v>=0?'#68d2ab':'#d56d85'),
   line:{color:'#0e2336',width:.7},opacity:state.points==='all'?.72:1},name:'Echte Strategien',showlegend:false};
  if(is3){args.z=dots.map(a=>a.v+.6);args.type='scatter3d';args.mode='markers';}
  else{args.type='scatter';args.mode='markers';}
  traces.push(args);
 }
 const vals=g.z.flat().filter(Number.isFinite); const bg='#102034',zrange=[Math.min(-50,...vals),Math.max(80,...vals)];
 const l=is3?{paper_bgcolor:bg,plot_bgcolor:bg,font:{color:'#c9d9ed',size:12},margin:{l:0,r:0,t:24,b:0},uirevision:'camera',showlegend:false,
  scene:{xaxis:x,yaxis:y,zaxis:{title:'Rendite 1× (%)',gridcolor:'#304b61',backgroundcolor:'#102437',range:zrange},camera:{eye:{x:1.68,y:-1.55,z:.86}}}}:
  {paper_bgcolor:bg,plot_bgcolor:bg,font:{color:'#c9d9ed',size:12},margin:{l:78,r:50,t:16,b:72},uirevision:'axes',showlegend:false,
   xaxis:{...x,gridcolor:'#243b52',zeroline:false,automargin:true},yaxis:{...y,gridcolor:'#243b52',zeroline:false,automargin:true}};
 return {traces,l};
}
function renderChart(){
 if(state.rendering){state.pending=true;return;}
 state.rendering=true;
 state.grid=buildHeatmap();
 const {traces,l}=makePlotData();
 Plotly.react($('chart'),traces,l,{responsive:true,scrollZoom:true,displaylogo:false}).then(()=>{
   if(!$('chart')._quanthandlers){$('chart')._quanthandlers=true;
    $('chart').on('plotly_click', ev=>{
      const pt=ev?.points?.[0];if(!pt)return;
      const raw=pt.customdata;
      if(Number.isInteger(raw)&&raw>=0&&META[raw]){inspect(raw);return;}
      const group=state.grid.groups.get(Number(pt.x)+'|'+Number(pt.y));
      if(group?.length){let best=group[0];for(const i of group)if(value(i)>value(best))best=i;inspect(best);}
    });
   }
 }).catch(err=>{console.error(err);if(state.chart==='3d'){state.chart='2d';$('chartType').value='2d';state.pending=true;}else $('plotCountNote').textContent='Grafikfehler: '+err.message;})
 .finally(()=>{state.rendering=false;if(state.pending){state.pending=false;renderChart();}});
}
function setDay(n){state.day=Math.max(0,Math.min(DAYS-1,n));state.grid=buildHeatmap();setStatus();renderChart();if(state.selected>=0)inspect(state.selected);}
function setControls(){
 $('axisX').value=state.x;$('axisY').value=state.y;$('exitFamily').value=state.family;
 $('boxFilter').value=state.box;$('poolMode').value=state.mode;$('ranking').value=String(state.rank);
 $('guard').value=state.guard;$('stopVal').textContent=state.guard===0?'Aus':'−'+state.guard+' %';
}
function changePool(){state.portKey='';state.filtered=eligible();state.grid=buildHeatmap();state.port=null;setStatus();renderChart();}
function applyPreset(n){
 if(n==='tp'){state.x='tp';state.y='s';state.family='tp';state.box='all';state.mode='all4158';}
 else if(n==='ttp'){state.x='act';state.y='s';state.family='ttp';state.box='all';state.mode='all4158';}
 else if(n==='tpttp'){state.x='tp';state.y='s';state.family='both';state.box='all';state.mode='all4158';}
 else if(n==='original'){state.x='s';state.y='b';state.family='none';state.box='all';state.priceMode='renko_usd';state.mode='original462';}
 else if(n==='all'){state.x='exit';state.y='s';state.family='all';state.box='all';state.mode='all4158';}
 else if(n==='top100'){state.x='s';state.y='b';state.family='all';state.box='all';state.mode='top100';}
 syncModeControls();setControls();changePool();
}
function serialize(){return {version:8,priceMode:state.priceMode,date:DATES[state.day],axisX:state.x,axisY:state.y,family:state.family,box:state.box,
 mode:state.mode,horizon:state.horizon,rank:state.rank,stop:state.guard,selected:[...state.manual].map(i=>META[i].id)};}
function downloadJSON(){const b=new Blob([JSON.stringify(serialize(),null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='quant_genome_v8_preset.json';a.click();URL.revokeObjectURL(u);}
function supportsWebGL(){try{const c=document.createElement('canvas');return !!(c.getContext('webgl2')||c.getContext('webgl')||c.getContext('experimental-webgl'));}catch(e){return false;}}

function modeLabel(){return state.priceMode==='candle'?'5-Minuten-Candles (kein Renko)':state.priceMode==='renko_pct'?'Renko · Prozent vom Kurs':'Renko · feste USD-Box';}
function syncModeControls(){
 $('useCandles').checked=state.priceMode==='candle';$('usePercent').checked=state.priceMode==='renko_pct';
 $('modeTag').textContent=modeLabel();
 $('modeExplanation').textContent=state.priceMode==='candle'?'Normaler 5m-OHLC-Kerzen-IMBA: S×10 Candles, keine Boxgröße.':'Alle tatsächlich berechneten Renko-Boxen verfügbar. Einzeln per Regler oder alle zusammen.';
 $('useAllBoxes').disabled=state.priceMode==='candle';
 if(state.priceMode==='candle')state.box='all';
 $('useAllBoxes').checked=state.box==='all';
 $('boxStep').disabled=state.priceMode==='candle'||state.box==='all';
 $('boxStepLabel').textContent=state.priceMode==='candle'?'kein Renko':(Number($('boxStep').value)/10).toFixed(1)+(state.priceMode==='renko_pct'?' %':' USD');
 const boxes=state.priceMode==='renko_pct'?Array.from({length:10},(_,i)=>(i+1)/10):
    state.priceMode==='candle'?[]:[...new Set(META.filter(m=>m.priceMode==='renko_usd').map(m=>m.b))].sort((a,b)=>a-b);
 const prev=state.box;
 populateSelector('boxFilter',[['all','Alle gerechneten Werte'],...boxes.map(b=>[b.toFixed(2),b.toFixed(2)+(state.priceMode==='renko_pct'?' %':' USD')])], prev);
 if($('boxFilter').value!==prev){state.box='all';$('boxFilter').value='all';$('useAllBoxes').checked=true;}
 if(state.x==='b'&&state.priceMode==='candle'){state.x='tp';state.y='s';$('axisX').value=state.x;$('axisY').value=state.y;}
}
function switchPriceMode(kind){
 if(!['candle','renko_pct','renko_usd'].includes(kind))return;
 state.priceMode=kind; state.box='all';
 if(state.dataset==='2022'&&kind!=='renko_usd'){$('dataset').value='2026';state.dataset='2026';}
 if(kind==='candle' && (state.x==='b'||state.y==='b')){state.x='tp';state.y='s';}
 syncModeControls();setControls();changePool();
}

function bindUI(){
 initOptions();
 syncModeControls();
 $('useCandles').onchange=e=>switchPriceMode(e.target.checked?'candle':'renko_usd');
 $('usePercent').onchange=e=>switchPriceMode(e.target.checked?'renko_pct':'renko_usd');
 $('useAllBoxes').onchange=e=>{state.box=e.target.checked?'all':(Number($('boxStep').value)/10).toFixed(2);syncModeControls();changePool();};
 $('boxStep').oninput=e=>{state.box=(Number(e.target.value)/10).toFixed(2);$('boxStepLabel').textContent=(Number(e.target.value)/10).toFixed(1)+(state.priceMode==='renko_pct'?' %':' USD');};
 $('boxStep').onchange=()=>{state.box=(Number($('boxStep').value)/10).toFixed(2);$('useAllBoxes').checked=false;syncModeControls();changePool();};
 if(!supportsWebGL()){state.chart='2d';$('chartType').value='2d';}
 $('datePrev').onclick=()=>setDay(state.day-1);$('dateNext').onclick=()=>setDay(state.day+1);
 $('slider').oninput=e=>setDay(+e.target.value);
 function stop(){if(state.playing!==null){clearInterval(state.playing);state.playing=null;}}
 $('pause').onclick=stop;
 function play(dir){stop();state.dir=dir;state.playing=setInterval(()=>{const nx=state.day+dir;if(nx<0||nx>=DAYS){stop();return;}setDay(nx);},Math.max(100,Number($('speed').value)));}
 $('playFwd').onclick=()=>play(1);$('playRev').onclick=()=>play(-1);
 $('presetTP').onclick=()=>applyPreset('tp');$('presetTTP').onclick=()=>applyPreset('ttp');
 $('presetBoth').onclick=()=>applyPreset('tpttp');$('preset462').onclick=()=>applyPreset('original');
 $('presetAll').onclick=()=>applyPreset('all');$('preset100').onclick=()=>applyPreset('top100');
 for(const [el,field] of [['axisX','x'],['axisY','y'],['exitFamily','family'],['boxFilter','box'],['poolMode','mode']])
  $(el).onchange=e=>{state[field]=e.target.value;if(field==='box'){if(state.box!=='all')$('boxStep').value=Math.round(Number(state.box)*10);syncModeControls();}if((field==='x'||field==='y')&&state.x===state.y){state[field==='x'?'y':'x']=field==='x'?'s':'b';}setControls();changePool();};
 $('dataset').onchange=e=>{state.dataset=e.target.value;if(state.dataset==='2022'&&state.priceMode!=='renko_usd')switchPriceMode('renko_usd');else changePool();};
 $('horizon').onchange=e=>{state.horizon=Number(e.target.value);state.grid=buildHeatmap();setStatus();renderChart();};
 $('aggregate').onchange=e=>{state.aggregate=e.target.value;state.grid=buildHeatmap();setStatus();renderChart();};
 $('chartType').onchange=e=>{state.chart=e.target.value;if(state.chart==='3d'&&!supportsWebGL()){state.chart='2d';$('chartType').value='2d';$('plotCountNote').textContent='WebGL nicht verfügbar – 2D-Heatmap aktiviert.';}renderChart();};
 $('pointsMode').onchange=e=>{state.points=e.target.value;renderChart();};
 $('ranking').onchange=e=>{state.rank=Number(e.target.value);changePool();};
 $('guard').oninput=e=>{$('stopVal').textContent=e.target.value==='0'?'Aus':'−'+e.target.value+' %';};
 $('guard').onchange=e=>{state.guard=Number(e.target.value);changePool();};
 $('geneScope').onchange=()=>renderGeneCards(choosePortfolio()[state.day]);
 $('manualAdd').onclick=()=>{if(state.selected>=0){state.manual.add(state.selected);inspect(state.selected);updateManual();if(state.mode==='manual')changePool();}};
 $('manualRemove').onclick=()=>{if(state.selected>=0){state.manual.delete(state.selected);inspect(state.selected);updateManual();if(state.mode==='manual')changePool();}};
 $('manualClear').onclick=()=>{state.manual.clear();updateManual();if(state.mode==='manual')changePool();};
 $('savePreset').onclick=downloadJSON;
 $('loadPreset').onchange=async e=>{const file=e.target.files[0];if(!file)return;
  try{const x=JSON.parse(await file.text());if(![7,8].includes(x.version))throw Error('Bitte eine v7/v8 Preset-Datei wählen.');
   state.manual=new Set((x.selected||[]).map(v=>IDMAP.get(v)).filter(Number.isInteger));
   if(DATES.includes(x.date))state.day=DATES.indexOf(x.date);
   state.x=AXES[x.axisX]?x.axisX:'tp';state.y=AXES[x.axisY]?x.axisY:'s';
   state.family=['all','none','tp','ttp','both'].includes(x.family)?x.family:'tp';
   state.priceMode=['candle','renko_pct','renko_usd'].includes(x.priceMode)?x.priceMode:'renko_usd';
   state.box=x.box||'all';state.mode=x.mode||'all4158';state.horizon=x.horizon??30;
   state.rank=x.rank??14;state.guard=x.stop??30;
   syncModeControls();setControls();$('horizon').value=state.horizon;$('slider').value=state.day;updateManual();changePool();
  }catch(err){alert('Preset konnte nicht geladen werden: '+err.message);}
 };
 $('downloadPool').onclick=()=>{const lines=['strategy_id,price_mode,sensitivity,renko_box_value,exit_program,tp_pct,ttp_activation_pct,ttp_trail_pct,return_2026_ytd_pct,return_2022_pct'];
  for(const i of state.filtered){const m=META[i];lines.push([m.id,m.priceMode,m.s,m.b,m.exitKey,m.tpValue,m.act,m.trail,rPct(i,DAYS-1,0),P22[i]].join(','));}
  const b=new Blob([lines.join('\n')],{type:'text/csv'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='quant_genome_v8_echte_kandidaten.csv';a.click();URL.revokeObjectURL(u);
 };
 state.points='top6';state.filtered=eligible();state.grid=buildHeatmap();
 setControls();setStatus();renderChart();updateManual();
}
function updateManual(){$('manualCount').textContent=state.manual.size+' individuell markierte Strategien';}
bindUI();