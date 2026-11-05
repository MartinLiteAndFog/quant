"""Generate a fully offline 2022–2026 Genome Lab with independently overlaid family surfaces."""
from pathlib import Path
import json,re,zipfile, numpy as np, pandas as pd
P=Path('/mnt/data');W=P/'imba_genome_v9';V8=P/'imba_genome_v8'
html=(P/'imba_2026_genome_parameter_lab_v8_renko_candles.html').read_text()
js=(V8/'app_v8.js').read_text()
payload=json.loads((W/'history_payload.json').read_text())
lengths={'2022':365,'2023':365,'2024':366,'2025':365,'2026':308}
# HTML: original dark design retained with prominent layers and year selection.
html=html.replace('Parameter Lab v8','Parameter Lab v9').replace('Genome Parameter Lab v8','Genome Parameter Lab v9').replace('>v8</span>','>v9</span>')
html=html.replace('2026: 01.01.–04.11.', '2022–2026 · Taggenaue Zeitreise')
html=html.replace('<option value="2026">2026 · Zeitfenster</option><option value="2022">2022 · Archiv: nur USD-Renko</option>',
 '''<option value="2022">2022 · 365 Tage</option><option value="2023">2023 · 365 Tage</option><option value="2024">2024 · 366 Tage</option><option value="2025">2025 · 365 Tage</option><option value="2026" selected>2026 · bis 04.11.</option>''')
html=html.replace('Seit 01.01.2026', 'Seit Jahresbeginn')
html=html.replace('Alle im Marktmodus · Exit × Sens','Vier Familien · S × Renko')
html=html.replace('10× BÜSCHEL / 2026 · AKTUELLER TAG','10× BÜSCHEL / GEWÄHLTES JAHR')
html=html.replace('style="color:#6fcbd5">v8','style="color:#6fcbd5">v9')
html=html.replace('2026: Neu berechneter', 'Fünf Jahre: Neu berechneter')
# Four surface controls below year/time bar; separate from candidate filters
layerbar='''<section class="bar" id="layerBar" style="align-items:center;border-color:#41627d;background:linear-gradient(95deg,#0d2033,#133047)">
 <div class="item"><label>OBERFLÄCHEN-MODUS</label><select id="surfaceMode"><option value="overlay" selected>Vier Familien übereinander</option><option value="combined">Eine gemeinsame Fläche</option></select></div>
 <label class="surfaceLayer"><input type="checkbox" data-layer="none" checked><span style="color:#75bafa">●</span> Ohne TP/TTP</label>
 <label class="surfaceLayer"><input type="checkbox" data-layer="tp" checked><span style="color:#f2af66">●</span> TP</label>
 <label class="surfaceLayer"><input type="checkbox" data-layer="ttp" checked><span style="color:#65d7ad">●</span> TTP</label>
 <label class="surfaceLayer"><input type="checkbox" data-layer="both" checked><span style="color:#cf9bff">●</span> TP+TTP</label>
 <div class="item"><label>TRANSPARENZ DER FLÄCHEN</label><input id="surfaceOpacity" type="range" min="20" max="90" step="5" value="62" class="range" style="width:145px"></div>
 <span class="muted" id="layerNote" style="max-width:360px">Vier unabhängig berechnete Flächen. Top-6-Kandidaten legt der Gen-Pool fest.</span>
</section>'''
html=html.replace('<div class="btnrow"><button id="presetTP">',layerbar+'\n<div class="btnrow"><button id="presetTP">')
html=html.replace('<section class="bar timebar">','<section class="bar timebar">')
# Time label in current HUD, explain year end resets
html=html.replace('Die Fläche aggregiert nur vorhandene Backtests; Portfolioauswahl verwendet ausschließlich vergangene Renditen.',
'Jede Exit-Familie besitzt ihre eigene Kurve im 3D-Raum. Ein Auswahlfilter für das Exit-Gen verändert die Top-6-Kandidaten; Flächen-Häkchen steuern nur die Sichtbarkeit. Jedes Jahr beginnt separat mit 1.000 €.')
html=html.replace('2022: Nur ursprünglicher USD-Renko-Sweep verfügbar.', '2022–2026: Alle Modi haben eigene Tagesreihen.')
html=html.replace('.range{accent-color:#55cdd5}', '.range{accent-color:#55cdd5}.surfaceLayer{display:inline-flex;align-items:center;gap:5px;background:#10283d;border:1px solid #2c536c;border-radius:8px;padding:8px 10px;cursor:pointer;white-space:nowrap;font-size:12px}.surfaceLayer input{accent-color:#5ec9b2}.surfaceLayer span{font-size:14px}')
# update backtest limits on sidebar
html=html.replace('2022er-Sweep verwendet ebenfalls 0,06 % Gebühren, aber eine andere Exit-Ausführung. Datenreihen nicht als identische Live-Parität lesen.',
'Für alle fünf Jahre gilt dasselbe 5-Minuten-Schlusskurs-Forschungsmodell mit 0,06 % Gebühren je Orderseite. Die 462er-Originaldaten aus älteren Studien sind nicht in jedem Jahr signalparitätisch. Ein Jahrestest beginnt jeweils mit 1.000 €; keine Kontofortschreibung über die Jahresgrenzen.')
html=html.replace('Der neue 4.158er-Sweep', 'Der 7.326er-Sweep')
html=html.replace('Rendite 2026er', 'Rendite 2022–2026er')
# Replace old script entire block to eliminate redundant F_B64/P22_B64 old data (~16MB)
start=html.index('<script>\nconst META=')
html=html[:start] + '''<script>
const META=%%META%%;
const BOXES=%%BOXES%%;
const ORIGINAL=%%ORIGINAL%%;
const ALL=%%ALL%%;
const RAW100=%%RAW100%%;
const BAL100=%%BAL100%%;
const YEAR_GZIP=%%YEAR_GZIP%%;
const YEAR_LENGTHS=%%YEAR_LENGTHS%%;
%%APPJS%%
</script></body></html>'''
# retrieve metadata/presets from original v8 embedded HTML via regular expressions
old=(P/'imba_2026_genome_parameter_lab_v8_renko_candles.html').read_text()
def extract(name):
    pattern=r'const '+name+r'=(.*?);\n'
    match=re.search(pattern,old)
    if not match:raise Exception('missing metadata '+name)
    return match.group(1)
meta=json.loads(extract('META'))
assert len(meta)==7326
# Initial year loading, no static single-year array
first=js.index('const N=META.length')
last=js.index('function dayFactor(i,d)')
replacement='''const N=META.length;
let DAYS=0, W=0, DATES=[], F=null, PREFIX=null;
const FAMILY_COLORS={none:'#75bafa',tp:'#f2af66',ttp:'#65d7ad',both:'#cf9bff'};
const GENE_ORDER=[['imba','⌁','IMBA','#48cbd5'],['renko','▦','RENKO','#70a8ed'],['candle','▤','CANDLES','#78d3ef'],['tp','↗','TP','#ecb56e'],['ttp','↝','TTP','#b995ef']];
const EXIT_NAMES=['Ohne TP/TTP','TP 6 %','TP 4 %','TP 3 %','TP6 + TTP3/1,5','TP 2 %','TP4 + TTP2/1','TTP 3/1,5','TTP 2/1'];
const AXES={s:'Sensitivity',b:'Renko-Box (USD)',tp:'TP-Ziel (%)',act:'TTP-Aktivierung (%)',trail:'TTP-Abstand (%)',exit:'Exit-Programm'};
const state={priceMode:'renko_usd',day:0,horizon:30,x:'s',y:'b',mode:'all4158',family:'all',box:'all',dataset:'2026',
  aggregate:'max',chart:'3d',points:'top6',guard:30,rank:14,manual:new Set(),selected:-1,
  playing:null,dir:1,port:null,portKey:'',filtered:[],gridN:0,cacheKey:'',grid:null,rendering:false,pending:false,
  surfaceMode:'overlay',layers:{none:true,tp:true,ttp:true,both:true},layerOpacity:0.62};
const IDMAP=new Map(META.map((m,i)=>[m.id,i]));
function decodeF32(b64){const raw=atob(b64), bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Float32Array(bytes.buffer);}
async function loadYear(year){
  if(!YEAR_GZIP[year])throw Error('Keine Backtest-Daten für '+year);
  if(typeof DecompressionStream!=='function')throw Error('Dieser Browser kann gzip-Daten nicht entpacken; bitte aktuelles Chrome verwenden.');
  const b64=YEAR_GZIP[year],raw=atob(b64), bytes=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  const arrbuf=await new Response(stream).arrayBuffer();
  DAYS=YEAR_LENGTHS[year];W=90+DAYS;
  if(arrbuf.byteLength!==N*W*4)throw Error(`Jahresdaten fehlerhaft ${year}: ${arrbuf.byteLength} Bytes`);
  F=new Float32Array(arrbuf);
  const start=new Date(Date.UTC(Number(year),0,1));
  DATES=Array.from({length:DAYS},(_,i)=>{const d=new Date(start.getTime()+i*86400000);return d.toISOString().slice(0,10);});
  PREFIX=new Float64Array(N*(W+1));
  for(let i=0;i<N;i++){const src=i*W,dst=i*(W+1);for(let j=0;j<W;j++)PREFIX[dst+j+1]=PREFIX[dst+j]+Math.log(Math.max(F[src+j],1e-15));}
  state.dataset=year;state.day=DAYS-1;state.portKey='';state.port=null;state.selected=-1;
  $('slider').max=String(DAYS-1);$('slider').value=String(state.day);$('dataset').value=year;
  $('inspectTitle').textContent='Heatmap-Zelle oder Bot anklicken';
}
'''
js=js[:first]+replacement+js[last:]
# date handling previous rolling
js=js.replace("function value(i){return state.dataset==='2022'?(META[i].has2022?P22[i]:NaN):rPct(i,state.day,state.horizon);}",
              "function value(i){return rPct(i,state.day,state.horizon);}")
js=js.replace("(state.dataset!=='2022'||META[i].has2022) && ", "")
js=js.replace('2026 YTD: ${pct(rPct(id,state.day,0))}', '${state.dataset} YTD: ${pct(rPct(id,state.day,0))}')
js=js.replace(" $('dataWarning').textContent=state.dataset==='2022'?'2022: Nur ursprünglicher USD-Renko-Sweep verfügbar. Prozent-Renko und Candles wurden nur 2025/2026 neu berechnet. Portfolio bleibt 2026.':'2026: Alle sichtbaren Bot-Kurven wurden für den jeweils gewählten Marktmodus berechnet (5m, Forschungsmodell; keine TradingView-Parität).';",
" $('dataWarning').textContent=`${state.dataset}: Echte Tagesrenditen für alle 7.326 Gen-Kombinationen im gewählten Marktmodus. 10× Portfolio mit eigenem Startkapital zu Jahresbeginn. Keine TradingView-/Liquidationsparität.`;")
js=js.replace(" $('plotCountNote').textContent=", " $('plotCountNote').textContent=")
# no switching away from 2022 anymore
js=js.replace(" if(state.dataset==='2022'&&kind!=='renko_usd'){$('dataset').value='2026';state.dataset='2026';}\n", "")
js=js.replace(" else if(n==='all'){state.x='exit';state.y='s';state.family='all';state.box='all';state.mode='all4158';}",
" else if(n==='all'){state.x='s';state.y='b';state.family='all';state.box='all';state.mode='all4158';state.surfaceMode='overlay';$('surfaceMode').value='overlay';}")
js=js.replace("function serialize(){return {version:8,priceMode:state.priceMode,date:DATES[state.day]", "function serialize(){return {version:9,priceMode:state.priceMode,date:DATES[state.day],year:state.dataset,surfaceMode:state.surfaceMode,layers:state.layers,layerOpacity:state.layerOpacity")
js=js.replace("a.download='quant_genome_v8_preset.json'", "a.download='quant_genome_v9_preset.json'")
js=js.replace("a.download='quant_genome_v8_echte_kandidaten.csv'", "a.download='quant_genome_v9_echte_kandidaten.csv'")
# implement layer grid per family; only groups containing actually computed bots become surfaces
anchor='function choosePortfolio(){'
assert anchor in js
layerFn='''function buildFamilyHeatmap(fam){
 const xs=state.grid.xs,ys=state.grid.ys;
 const groups=new Map();
 for(const i of state.filtered){if(META[i].family!==fam)continue;
   const key=realAxisValue(i,state.x)+'|'+realAxisValue(i,state.y);
   if(!groups.has(key))groups.set(key,[]);groups.get(key).push(i);
 }
 const z=[],info=[],custom=[];let occupied=0;const seenX=new Set(),seenY=new Set();
 for(const y of ys){const rz=[],ri=[],rc=[];
   for(const x of xs){const ids=groups.get(x+'|'+y)||[];
     if(!ids.length){rz.push(null);ri.push('Keine getestete Kombination');rc.push(null);continue;}
     seenX.add(x);seenY.add(y);occupied++;
     let sum=0,best=ids[0],mx=-Infinity;
     for(const i of ids){const v=value(i);sum+=v;if(v>mx){mx=v;best=i;}}
     const v=state.aggregate==='max'?mx:sum/ids.length;
     rz.push(v);rc.push(best);
     ri.push(`${familyName(fam)} · ${ids.length} echte Strategien<br>Rendite ${pct(v)}<br>Bester Bot ${META[best].id}<br>Bestwert ${pct(mx)}`);
   }
   z.push(rz);info.push(ri);custom.push(rc);
 }
 return {xs,ys,z,info,custom,occupied,seenX,seenY};
}
'''
js=js.replace(anchor,layerFn+anchor)
# Overlay implementation replace entire makePlotData
left=js.index('function makePlotData(){')
right=js.index('function renderChart(){',left)
plot='''function makePlotData(){const g=state.grid,{x,y}=layoutAxes(),is3=state.chart==='3d';
 const traces=[];
 const scale=[[0,'#ca596c'],[.38,'#75465f'],[.5,'#23455f'],[.65,'#26796f'],[1,'#65cea7']];
 let overlayShapes=0,degenerate=[];
 const overlay=is3&&state.surfaceMode==='overlay'&&state.family==='all';
 if(overlay){
   for(const fam of ['none','tp','ttp','both']){
     if(!state.layers[fam])continue;
     const layer=buildFamilyHeatmap(fam);
     if(!layer.occupied)continue;
     const color=FAMILY_COLORS[fam];
     if(layer.seenX.size>=2&&layer.seenY.size>=2){
       traces.push({type:'surface',x:layer.xs,y:layer.ys,z:layer.z,
         colorscale:[[0,color],[1,color]],opacity:state.layerOpacity,connectgaps:false,
         customdata:layer.custom,hovertext:layer.info,hoverinfo:'text',showscale:false,
         showlegend:false,name:familyName(fam)});
       overlayShapes++;
     }else{
       // A 1D or singleton family cannot form an actual 2D surface; show the real cells as line/dots.
       degenerate.push(familyName(fam));
       const dots=[];
       layer.z.forEach((row,iy)=>row.forEach((v,ix)=>{if(v!==null&&Number.isFinite(v))dots.push([layer.xs[ix],layer.ys[iy],v,layer.info[iy][ix],layer.custom[iy][ix]]);}));
       dots.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
       traces.push({type:'scatter3d',mode:'markers+lines',x:dots.map(d=>d[0]),y:dots.map(d=>d[1]),z:dots.map(d=>d[2]),
         customdata:dots.map(d=>d[4]),hovertext:dots.map(d=>d[3]),hoverinfo:'text',
         marker:{color,size:5,opacity:.94},line:{color,width:4},name:familyName(fam),showlegend:false});
     }
   }
 }else if(g.xs.length&&g.ys.length){
   if(is3&&g.xs.length>=2&&g.ys.length>=2){
     traces.push({type:'surface',x:g.xs,y:g.ys,z:g.z,colorscale:scale,opacity:.90,connectgaps:false,
       customdata:g.custom,hovertext:g.info,hoverinfo:'text',name:'Getestete Varianten',showscale:true,colorbar:{title:'1× %',len:.48,thickness:11}});
   }else if(!is3){
     traces.push({type:'heatmap',x:g.xs,y:g.ys,z:g.z,colorscale:scale,connectgaps:false,
       customdata:g.custom,hovertext:g.info,hoverinfo:'text',name:'Getestete Varianten',showscale:true,colorbar:{title:'1× %',thickness:11}});
   }
 }
 const note=$('layerNote');
 if(note)note.textContent=overlay
   ? `${overlayShapes} separate Flächen${degenerate.length?', als Linien/Punkte: '+degenerate.join(', '):''} · Sichten unabhängig · Nur echte Gen-Kombinationen`
   : (is3?'Einzeloberfläche · aktives Exit-Gen grenzt den Kandidatenpool ein':'2D: Aggregierte Heatmap der gewählten Familien');
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
 const vals=g.z.flat().filter(Number.isFinite), bg='#102034',zrange=[Math.min(-50,...vals),Math.max(80,...vals)];
 const l=is3?{paper_bgcolor:bg,plot_bgcolor:bg,font:{color:'#c9d9ed',size:12},margin:{l:0,r:0,t:24,b:0},uirevision:'camera',showlegend:false,
  scene:{xaxis:x,yaxis:y,zaxis:{title:'Rendite 1× (%)',gridcolor:'#304b61',backgroundcolor:'#102437',range:zrange},camera:{eye:{x:1.68,y:-1.55,z:.86}}}}:
  {paper_bgcolor:bg,plot_bgcolor:bg,font:{color:'#c9d9ed',size:12},margin:{l:78,r:50,t:16,b:72},uirevision:'axes',showlegend:false,
   xaxis:{...x,gridcolor:'#243b52',zeroline:false,automargin:true},yaxis:{...y,gridcolor:'#243b52',zeroline:false,automargin:true}};
 return {traces,l};
}
'''
js=js[:left]+plot+js[right:]
# years switching and UI controls
js=js.replace(" $('dataset').onchange=e=>{state.dataset=e.target.value;if(state.dataset==='2022'&&state.priceMode!=='renko_usd')switchPriceMode('renko_usd');else changePool();};",
" $('dataset').onchange=async e=>{const year=e.target.value;try{$('dataset').disabled=true;$('dataWarning').textContent='Lade vollständige Tageskurven für '+year+' …';if(state.playing!==null){clearInterval(state.playing);state.playing=null;}await loadYear(year);changePool();}catch(err){console.error(err);alert('Jahr konnte nicht geladen werden: '+err.message);}finally{$('dataset').disabled=false;}};")
# add controls, family change when overlay auto axes to common 2D parameters if necessary
js=js.replace(" $('pointsMode').onchange=e=>{state.points=e.target.value;renderChart();};",
" $('pointsMode').onchange=e=>{state.points=e.target.value;renderChart();};\n $('surfaceMode').onchange=e=>{state.surfaceMode=e.target.value;renderChart();};\n $('surfaceOpacity').oninput=e=>{state.layerOpacity=Number(e.target.value)/100;renderChart();};\n document.querySelectorAll('[data-layer]').forEach(el=>el.onchange=e=>{state.layers[e.target.dataset.layer]=e.target.checked;renderChart();});")
js=js.replace("function serialize(){return {version:9,priceMode", "function serialize(){return {version:9,priceMode")
js=js.replace(" if(![7,8].includes(x.version))", " if(![7,8,9].includes(x.version))")
js=js.replace("throw Error('Bitte eine v7/v8 Preset-Datei wählen.')", "throw Error('Bitte eine v7/v8/v9 Preset-Datei wählen.')")
js=js.replace("   if(DATES.includes(x.date))state.day=DATES.indexOf(x.date);",
"   if(x.year&&String(x.year)!==state.dataset){await loadYear(String(x.year));}else if(x.date&&!DATES.includes(x.date)&&YEAR_LENGTHS[String(x.date).slice(0,4)])await loadYear(String(x.date).slice(0,4));\n   if(DATES.includes(x.date))state.day=DATES.indexOf(x.date);\n   if(x.surfaceMode)state.surfaceMode=x.surfaceMode;if(x.layers)Object.assign(state.layers,x.layers);if(x.layerOpacity)state.layerOpacity=x.layerOpacity;\n   $('surfaceMode').value=state.surfaceMode;$('surfaceOpacity').value=String(Math.round(state.layerOpacity*100));\n   document.querySelectorAll('[data-layer]').forEach(el=>el.checked=!!state.layers[el.dataset.layer]);")
# remove P22 in export
js=js.replace("'return_2026_ytd_pct,return_2022_pct'", "'return_'+state.dataset+'_ytd_pct'")
js=js.replace("rPct(i,DAYS-1,0),P22[i]", "rPct(i,DAYS-1,0)")
# remove 2022-only switch condition in switchPriceMode (already)
# initialize async; only render after loading data
assert js.strip().endswith('bindUI();')
js=js.rsplit('bindUI();',1)[0]+'''async function startApp(){try{await loadYear('2026');bindUI();window.__QUANT_V9_READY=true;}catch(e){console.error('Quant v9 failed to load',e);alert('Daten konnten nicht geladen werden: '+e.message);}}\nstartApp();\n'''
# year string in method
js=js.replace("function rPct(i,end,window){", "function rPct(i,end,window){")
js=js.replace("const p=90+end+1,left=window?Math.max(0,p-window):90;", "const p=90+end+1,left=window?Math.max(0,p-window):90;")
js=js.replace("$('inspectReturn').textContent=`2026 YTD:", "$('inspectReturn').textContent=`${state.dataset} YTD:")
js=js.replace("state.portKey=hash;state.port=results", "state.portKey=hash;state.port=results")
# normalize dataset version and box modes
js=js.replace("if(state.dataset==='2022'&&kind!=='renko_usd'){$('dataset').value='2026';state.dataset='2026';}","")
# initial filtered / behavior setControl okay
# append data and JS
subs={
 '%%META%%':extract('META'),'%%BOXES%%':extract('BOXES'),'%%ORIGINAL%%':extract('ORIGINAL'),
 '%%ALL%%':extract('ALL'),'%%RAW100%%':extract('RAW100'),'%%BAL100%%':extract('BAL100'),
 '%%YEAR_GZIP%%':json.dumps(payload,separators=(',',':')),
 '%%YEAR_LENGTHS%%':json.dumps(lengths,separators=(',',':')),
 '%%APPJS%%':js,
}
for k,v in subs.items():
 assert k in html, k
 html=html.replace(k,v)
assert not re.findall(r'%%[A-Z_]+%%',html), re.findall(r'%%[A-Z_]+%%',html)
(W/'app_v9.js').write_text(js)
out=P/'imba_genome_parameter_lab_v9_2022_2026.html'
out.write_text(html)
print('Built',out,'MB',round(out.stat().st_size/1e6,2),'metadata',len(meta),'days',sum(lengths.values()))