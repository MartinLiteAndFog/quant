"""Build Genome Parameter Lab v13 over verified v12 snapshot.
Preserves 7,326 actual daily strategy backtests, adds 2026 named screenshot
presets, annual training family selectors, and evidence-based interactive charts.
"""
from pathlib import Path
import json, pandas as pd
ROOT=Path('/mnt/data');OUT=ROOT/'imba_genome_v13_work'
src=(ROOT/'imba_genome_parameter_lab_v12_familien_transparent.html').read_text('utf-8')
app=(ROOT/'imba_genome_app_v12.js').read_text('utf-8')
assert src.count(app)==1

# Immutable training-year top 100 by original strategy index; same META universe.
annual=pd.read_csv(ROOT/'imba_annual_raw100_2022_2026/annual_top100_raw_families_2022_2025.csv')
vintages={str(int(y)):g.sort_values('rank').bot_index.astype(int).tolist() for y,g in annual.groupby('training_year')}
assert all(len(v)==100 for v in vintages.values())

shots={
 'original_2026_14d_20':{'label':'Original 462 · 14 Tage · Einzelstopp 20 %','pool':'original462','rank':14,'guard':20,'screenshot':'Screenshot 2 (09:41:04)','wealth_at_1000':9367.04925136168,'verified':True},
 'balanced_2026_30d_30':{'label':'Top100 Balanced · 30 Tage · Einzelstopp 30 %','pool':'balanced100','rank':30,'guard':30,'screenshot':'Screenshot 1 (09:39:49)','wealth_at_1000':7242.873165803406,'verified':True},
 'balanced_2026_30d_25':{'label':'Top100 Balanced · 30 Tage · Einzelstopp 25 %','pool':'balanced100','rank':30,'guard':25,'screenshot':'Screenshot 3 (09:41:58)','wealth_at_1000':10654.973910559,'verified':True},
 'raw_2026_30d_30_comparison':{'label':'Top100 Raw · 30 Tage · Einzelstopp 30 % (Vergleich)','pool':'top100','rank':30,'guard':30,'screenshot':None,'wealth_at_1000':3493.6135685711342,'verified':True,'note':'Keiner der drei Screenshots zeigt Raw; Vergleichsannahme aus früheren Tests.'},
}
(OUT/'screenshots_2026_presets.json').write_text(json.dumps(shots,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

# JS patch - no change to portfolio ledger or its mathematics.
needle='const IDMAP=new Map(META.map((m,i)=>[m.id,i]));'
assert app.count(needle)==1
app=app.replace(needle,needle+'\n'+'const YEAR_RAW_POOLS='+json.dumps(vintages,separators=(',',':'))+';\n'
    +'const SCREENSHOT_PRESETS='+json.dumps(shots,ensure_ascii=False,separators=(',',':'))+';')
needle=" if(state.mode==='manual')return [...state.manual];"
assert app.count(needle)==1
app=app.replace(needle," if(state.mode==='manual')return [...state.manual];\n"
                 " if(state.mode.startsWith('rawYear')){const vintage=state.mode.slice(7);return Number(state.dataset)>Number(vintage)?(YEAR_RAW_POOLS[vintage]||[]):[];}")
needle="const modeNames={original462:'Original 462',top100:'Top 100 aus 2022 · höchste Rendite',balanced100:'Top 100 aus 2022 · je 25 pro Exit-Familie',all4158:'Alle wirklich berechneten Strategien',grid20:'20×20 Parameter-Filter',grid30:'30×30 Parameter-Filter',manual:'Manuell gesetzte Bots'};"
assert app.count(needle)==1
app=app.replace(needle,needle[:-2]+",rawYear2022:'Raw Top100 · Training 2022',rawYear2023:'Raw Top100 · Training 2023',rawYear2024:'Raw Top100 · Training 2024',rawYear2025:'Raw Top100 · Training 2025'};")
needle="$('dataWarning').textContent=`${state.dataset}: Echte Tagesrenditen für alle 7.326 Gen-Kombinationen im gewählten Marktmodus. 10× Portfolio mit eigenem Startkapital zu Jahresbeginn. Keine TradingView-/Liquidationsparität.`;"
assert app.count(needle)==1
app=app.replace(needle,needle+"\n if(state.mode.startsWith('rawYear')&&Number(state.dataset)<=Number(state.mode.slice(7)))$('dataWarning').textContent+=' Trainingsfamilie darf erst in den folgenden Jahren gehandelt werden: kein Lookahead; deshalb keine Kandidaten.';")
needle="for(const [el,field] of [['axisX','x'],['axisY','y'],['boxFilter','box'],['poolMode','mode']])\n   $(el).onchange=e=>{state[field]=e.target.value;"
assert app.count(needle)==1
app=app.replace(needle,needle.replace('onchange=e=>','onchange=async e=>'))
needle="if(field==='mode'&&['original462','top100','balanced100'].includes(state.mode)&&state.priceMode!=='renko_usd'){"
assert app.count(needle)==1
app=app.replace(needle,"if(field==='mode'&&state.mode.startsWith('rawYear') && Number(state.dataset)<=Number(state.mode.slice(7))){\n       const targetYear=String(Number(state.mode.slice(7))+1);await loadYear(targetYear);}\n    "+needle.replace("['original462','top100','balanced100'].includes(state.mode)","(['original462','top100','balanced100'].includes(state.mode)||state.mode.startsWith('rawYear'))"))
# ensure report data is never used for trading selection (separate constant)
extra_js=r'''

// 2026 favorite settings from screenshots and evidence charts, NOT live trading changes.
let researchMode='mix';
let reportIsOpen=false;
function saved2026Info(key){
 const p=SCREENSHOT_PRESETS[key];const info=$('snapshotNote');
 if(!p){info.textContent='Bitte ein Preset wählen.';return;}
 const original=p.screenshot?`Gesichert aus ${p.screenshot}`:'Vergleichsvariante – nicht in den drei Screenshots enthalten';
 info.innerHTML=`<strong>${p.label}</strong><br>${original}. Originalwert bei 1.000 € bis 04.11.2026: <b>${euro(p.wealth_at_1000)}</b>.<br>Nur 2026 · fester USD-Renko · Auswahl alle 14 Tage · positiver Ranking-Score erforderlich.`;
}
async function loadSaved2026(key){
 const p=SCREENSHOT_PRESETS[key];if(!p)return;
 if(state.playing!==null){cancelAnimationFrame(state.playing);state.playing=null;}
 if(state.dataset!=='2026')await loadYear('2026');
 state.mode=p.pool;state.rank=p.rank;state.guard=p.guard;
 state.family='all';state.familyEnabled={none:true,tp:true,ttp:true,both:true};
 state.priceMode='renko_usd';state.box='all';state.x='s';state.y='b';state.day=DAYS-1;
 state.chart=supportsWebGL()?'3d':'2d';state.points='top6';state.horizon=30;
 $('chartType').value=state.chart;$('horizon').value=30;$('pointsMode').value='top6';
 $('slider').value=state.day;
 syncModeControls();setControls();changePool();saved2026Info(key);
 $('activeSnapshot').textContent=`Aktiv: ${p.label}`;
}
function chartTheme(title,extra={}){return {...{
 title:{text:title,font:{size:16,color:'#dcecf5'},x:0.03},paper_bgcolor:'#0c2336',plot_bgcolor:'#102b40',
 font:{family:'system-ui, Arial',color:'#c9deed',size:12},margin:{t:55,l:65,r:35,b:55},
 xaxis:{gridcolor:'#2b4758',zerolinecolor:'#335668',title:''},
 yaxis:{gridcolor:'#2b4758',zerolinecolor:'#335668',title:'Depotwert (€)',tickformat:',.0f'},
 legend:{orientation:'h',x:0,y:1.12,font:{size:10}},hovermode:'x unified',uirevision:'quant-v13-'+researchMode
 },...extra};}
const chartOpts={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};
function renderResearchCharts(){
 if(!reportIsOpen)return;
 const b=EVIDENCE_CHARTS;
 const names={original:'Original 462 · 333,33 €',raw:'Top100 Raw · 333,33 €',balanced:'Balanced B · 333,34 €',mixed:'Drei-Büschel-Mix',previous:'Bisheriger Mix (Balanced −30 %)'};
 const color={original:'#61bdd8',raw:'#f3ad71',balanced:'#c3a1f4',mixed:'#e7d27a',previous:'#8493a7'};
 const el=$('researchChart');
 const note=$('researchChartNote');
 if(researchMode==='mix'){
  const traces=['original','raw','balanced','mixed','previous'].map(k=>({type:'scatter',mode:'lines',x:b.mix.dates,y:b.mix[k],name:names[k],
   line:{color:color[k],width:k==='mixed'?3:k==='previous'?1.5:1.9,dash:k==='previous'?'dot':'solid'}}));
  Plotly.react(el,traces,chartTheme('2026 · Ein Depot, drei unabhängige Büschel'),chartOpts);
  note.textContent='Start: 333,33 € + 333,33 € + 333,34 € = 1.000 €. Der Mix ist die Summe der drei getrennt verwalteten Kapitalkurven. Die ausgepunktete Variante verwendet Balanced −30 % statt −25 %.';
 }else if(researchMode==='sharpe'){
  const stats=b.stats;const name=stats.map(s=>s.portfolio),cols=['#61bdd8','#f3ad71','#c3a1f4','#e7d27a'];
  const traces=[{type:'bar',name:'Sharpe (365d)',x:name,y:stats.map(v=>v.sharpe_365d),marker:{color:cols},text:stats.map(v=>v.sharpe_365d.toFixed(2)),textposition:'outside',hovertemplate:'%{x}: Sharpe %{y:.3f}<extra></extra>'}];
  Plotly.react(el,traces,chartTheme('2026 · Sharpe Ratio, einheitlich 1.000 € Startkapital',{
    margin:{t:55,l:60,r:25,b:115},xaxis:{tickangle:-18,gridcolor:'#2b4758'},yaxis:{title:'Sharpe · 365 Tage',gridcolor:'#2b4758'},showlegend:false}),chartOpts);
  note.innerHTML='<b>Endwerte / maximaler Drawdown:</b> '+stats.map(s=>`${s.portfolio}: ${euro(s.end_eur)} / ${s.max_drawdown_pct.toFixed(1)} %`).join(' · ')+'<br>Vergleich basiert auf täglichen Depotwertänderungen (risikofrei = 0, 365 Tage annualisiert).';
 }else{
  const family=b.annual;
  const tr=[['carry_2022','2022er Familie behalten','#f3ad71'],['annual_rotate','Jährlich neue Top 100','#70d6ab'],['accumulate_past','Frühere Jahrgänge sammeln','#c3a1f4']].map(([key,name,col])=>({
    type:'scatter',mode:'lines',x:family[key].dates,y:family[key].wealth,name,line:{color:col,width:2}}));
  Plotly.react(el,tr,chartTheme('2023–2026 · Top-100-Jahresfamilien: durchgehendes Kapital',{
    yaxis:{title:'Depotwert (€), logarithmisch',type:'log',gridcolor:'#2b4758',range:[-2,4.5]},
    legend:{orientation:'h',x:0,y:1.14,font:{size:10}}}),chartOpts);
  note.textContent='Separate Forschungsreihe: 1.000 € Anfang 2023, danach ohne Neustart, 14-Tage-Ranking, 30-%-Einzelstopps. Logarithmische Euro-Skala. Jahrgänge werden ausschließlich aus abgeschlossenen Vorjahren ausgewählt.';
 }
}
function bindSavedEvidence(){
 const selector=$('snapshotPreset');
 selector.onchange=e=>saved2026Info(e.target.value);
 $('applySnapshot').onclick=()=>loadSaved2026(selector.value);
 saved2026Info(selector.value);
 $('researchDetails').addEventListener('toggle',()=>{
  reportIsOpen=$('researchDetails').open;
  if(reportIsOpen)requestAnimationFrame(renderResearchCharts);
 });
 $('chartsJump').onclick=()=>{
  const panel=$('researchDetails');panel.open=true;panel.scrollIntoView({behavior:'smooth',block:'start'});
  reportIsOpen=true;requestAnimationFrame(renderResearchCharts);
 };
 for(const btn of document.querySelectorAll('[data-research-mode]')){
  btn.onclick=()=>{
   researchMode=btn.dataset.researchMode;
   document.querySelectorAll('[data-research-mode]').forEach(b=>b.classList.toggle('active',b===btn));
   renderResearchCharts();
  };
 }
}
'''
# insert JS functions after bindUI & before app entry (end)
needle='async function startApp(){try{await loadYear(\'2026\');bindUI();window.__QUANT_V12_READY=true;}'
assert app.count(needle)==1
app=app.replace(needle,extra_js+'\n'+needle.replace('bindUI();','bindUI();bindSavedEvidence();'))
# Correlation selector: causal 28-day Pearson cap in top-six portfolio selection.
# "strict" means at most six; optional fallback explicitly violates the cap.
needle="  aggregate:'max',chart:'3d',points:'top6',guard:30,rank:14,manual:new Set(),selected:-1,"
assert app.count(needle)==1
app=app.replace(needle,needle.replace('guard:30,rank:14,','guard:30,rank:14,corrLimit:null,corrFallback:false,'))
needle="const pool=state.filtered,hash=pool.join(',')+'|'+state.guard+'|'+state.rank;"
assert app.count(needle)==1
app=app.replace(needle,"const pool=state.filtered,hash=pool.join(',')+'|'+state.guard+'|'+state.rank+'|'+state.corrLimit+'|'+state.corrFallback;")
needle="const scored=pool.map(i=>[i,score(i,d)]).filter(v=>v[1]>0).sort((a,b)=>b[1]-a[1]||a[0]-b[0]).slice(0,6);"
assert app.count(needle)==1
app=app.replace(needle,"const allRanked=pool.map(i=>[i,score(i,d)]).filter(v=>v[1]>0).sort((a,b)=>b[1]-a[1]||a[0]-b[0]);\n   const scored=applyCorrelationCap(allRanked,d);")
needle='function choosePortfolio(){'
assert app.count(needle)==1
app=app.replace(needle,r'''function pearsonPrevious28(a,b,day){
  // 28 COMPLETE daily 1× returns before selection date; no current-day future prices.
  const start=90+day-28,end=90+day,n=28,baseA=a*W,baseB=b*W;
  let sx=0,sy=0,sxx=0,syy=0,sxy=0;
  for(let t=start;t<end;t++){
    const x=F[baseA+t]-1,y=F[baseB+t]-1;
    sx+=x;sy+=y;sxx+=x*x;syy+=y*y;sxy+=x*y;
  }
  const xx=sxx-sx*sx/n,yy=syy-sy*sy/n;
  if(xx<1e-18||yy<1e-18)return 0;
  return Math.max(-1,Math.min(1,(sxy-sx*sy/n)/Math.sqrt(xx*yy)));
}
function applyCorrelationCap(ranked,day){
 if(state.corrLimit===null||!Number.isFinite(state.corrLimit))return ranked.slice(0,6);
 const selected=[];
 for(const candidate of ranked){
   if(selected.length>=6)break;
   if(selected.every(s=>pearsonPrevious28(s[0],candidate[0],day)<=state.corrLimit))selected.push(candidate);
 }
 if(state.corrFallback&&selected.length<6){
   const chosen=new Set(selected.map(s=>s[0]));
   for(const c of ranked){if(selected.length>=6)break;if(!chosen.has(c[0])){selected.push(c);chosen.add(c[0]);}}
 }
 return selected;
}
''' +needle)
needle="$('guard').value=state.guard;$('stopVal').textContent=state.guard===0?'Aus':'−'+state.guard+' %';"
assert app.count(needle)==1
app=app.replace(needle,needle+"\n $('correlationLimit').value=state.corrLimit===null?'off':String(state.corrLimit);$('corrFallback').checked=state.corrFallback;")
needle="$('guard').onchange=e=>{state.guard=Number(e.target.value);changePool();};"
assert app.count(needle)==1
app=app.replace(needle,needle+"\n $('correlationLimit').onchange=e=>{state.corrLimit=e.target.value==='off'?null:Number(e.target.value);changePool();};\n $('corrFallback').onchange=e=>{state.corrFallback=e.target.checked;changePool();};")
needle='state.rank=x.rank??14;state.guard=x.stop??30;'
assert app.count(needle)==1
app=app.replace(needle,needle+"state.corrLimit=(x.corrLimit===null||x.corrLimit===undefined)?null:Number(x.corrLimit);state.corrFallback=!!x.corrFallback;")
needle='mode:state.mode,familyEnabled:state.familyEnabled,horizon:state.horizon,rank:state.rank,stop:state.guard,selected:'
assert app.count(needle)==1
app=app.replace(needle,needle.replace('rank:state.rank,stop:state.guard,','rank:state.rank,stop:state.guard,corrLimit:state.corrLimit,corrFallback:state.corrFallback,'))
needle="state.mode=p.pool;state.rank=p.rank;state.guard=p.guard;"
assert app.count(needle)==1
app=app.replace(needle,needle+"state.corrLimit=null;state.corrFallback=false;")
needle='renderGeneCards(d);renderTop6(d);renderEquityChart();renderFamilyBreakdown(d);'
assert app.count(needle)==1
app=app.replace(needle,needle+"\n const shot=$('activeSnapshot');if(shot){const match=Object.values(SCREENSHOT_PRESETS).find(x=>x.pool===state.mode&&x.guard===state.guard&&x.rank===state.rank&&state.dataset==='2026'&&state.priceMode==='renko_usd'&&state.box==='all'&&state.corrLimit===null&&Object.values(state.familyEnabled).every(Boolean));shot.textContent=match?'Aktiv: '+match.label:'Aktive Einstellungen: individuell verändert / anderes Jahr';}")
needle="$('plotCountNote').textContent=`${len} echte Bot-Kurven · ${state.grid.occupied} belegte Parameterzellen · ${state.priceMode==='candle'?'5-Minuten-Candles':state.priceMode==='renko_pct'?'Prozent-Renko':'USD-Renko'} · ${state.box==='all'?'Alle getesteten Größen': 'Box '+state.box} · 10× nur im Portfolio`;"
assert app.count(needle)==1
app=app.replace(needle,needle+"\n $('corrInfo').textContent=state.corrLimit===null?'Korrelationsfilter ausgeschaltet':('Pearson-Korrelation der letzten 28 vollständigen Tage ≤ '+state.corrLimit.toFixed(2)+(state.corrFallback?' · Aufstockung kann den Grenzwert verletzen':' · strikt, auch weniger als sechs Bots möglich'));")


app=app.replace('window.__QUANT_V12_READY','window.__QUANT_V13_READY').replace('Quant Genome v12','Quant Genome v13')
app=app.replace('version:12,priceMode','version:13,priceMode').replace('quant_genome_v12_preset.json','quant_genome_v13_preset.json').replace('[7,8,9,10,11,12]','[7,8,9,10,11,12,13]')
app=app.replace("console.error('Quant v9 failed to load'","console.error('Quant v13 failed to load'")
(OUT/'imba_genome_app_v13.js').write_text(app,encoding='utf8')

# HTML additions
src=src.replace('IMBA Parameter Lab <span style="color:#6fcbd5">v12</span>','IMBA Parameter Lab <span style="color:#6fcbd5">v13</span>')
src=src.replace('7.326 tatsächlich gerechnete Bot-Genkombinationen · transparentes Pool-Ranking',
                '7.326 echte Bot-Kombinationen · Jahresfamilien · gespeicherte 2026-Presets · Charts')
needle='<div class="workspaceSwitch"><button type="button" id="sidebarToggle">Seitenmenüs ausblenden</button>'
assert src.count(needle)==1
src=src.replace(needle,'<div class="workspaceSwitch"><button id="chartsJump" class="highlightCharts" type="button">Kapitalkurven & Sharpe ansehen</button><button type="button" id="sidebarToggle">Seitenmenüs ausblenden</button>')
# sidebar favorites (insert before right existing first panel)
needle='<aside class="right"><section class="panel familyAuditPanel">'
assert src.count(needle)==1
favs='''<aside class="right"><section class="panel savedPresets"><h2>2026 · Gespeicherte Favoriten</h2>
 <p class="muted">Drei Screenshots als exakt reproduzierte Kombinationen aus Familie, Top-6-Ranking und Einzelstopp. Die ersten beiden Balanced-Screenshots sind <strong>keine</strong> Raw-Top-100.</p>
 <div class="field"><span class="smallLabel">SCREENSHOT-PRESET</span><select id="snapshotPreset">
  <option value="original_2026_14d_20">Original 462 · Ranking 14d · Stopp 20 %</option>
  <option value="balanced_2026_30d_30">Balanced A · Ranking 30d · Stopp 30 %</option>
  <option value="balanced_2026_30d_25" selected>Balanced B · Ranking 30d · Stopp 25 %</option>
  <option value="raw_2026_30d_30_comparison">Raw100 · Ranking 30d · Stopp 30 % (Vergleich)</option>
 </select></div>
 <button type="button" id="applySnapshot" class="applySaved">Einstellung in App übernehmen</button>
 <div id="snapshotNote" class="savedNote"></div><div id="activeSnapshot" class="muted"></div>
 </section><section class="panel familyAuditPanel">'''
src=src.replace(needle,favs)
# annual families in pool dropdown, option after original balanced
needle='<option value="balanced100">2022 Top 100 Balanced</option>'
assert src.count(needle)==1
src=src.replace(needle,needle+''.join(f'<option value="rawYear{y}">Top100 Raw · Training {y} → Test ab {y+1}</option>' for y in [2022,2023,2024,2025]))
# Correlation controls affect the actual selection algorithm (not only visual display).
needle='<p class="muted">Nur positive Renditen im gewählten Rückblick · höchstens 6 aktive Bots'
assert src.count(needle)==1
corr_ui="""<div class="controlGrid" style="margin:9px 0"><div class="item"><label>KORRELATION (28 TAGE ZURÜCK)</label><select id="correlationLimit"><option value="off" selected>Aus · keine Grenze</option><option value="0.85">Maximal +0,85</option><option value="0.9">Maximal +0,90</option></select></div><div class="item"><label class="toggle"><input id="corrFallback" type="checkbox"> Bei zu wenig Bots auffüllen (bricht Grenze)</label></div></div><div id="corrInfo" class="muted">Korrelationsfilter ausgeschaltet</div>"""
src=src.replace(needle,corr_ui+needle)
# chart section below 3D panel before aside
needle='</p></div></div>\n<aside class="right">'
# we've modified aside right marker so find from original fine print note
needle='Jedes Jahr beginnt separat mit 1.000 €.</p></div></div>\n<aside class="right">'
# check
assert src.count(needle)==1,src.count(needle)
charts='''Jedes Jahr beginnt separat mit 1.000 €.</p></div>
<details id="researchDetails" class="researchPanel"><summary>Forschungs-Charts · 2026er Büschelmix, Sharpe &amp; Jahresfamilien</summary>
 <div class="researchInner"><div class="researchControls">
  <button type="button" data-research-mode="mix" class="active">2026 · Kapital</button>
  <button type="button" data-research-mode="sharpe">2026 · Sharpe</button>
  <button type="button" data-research-mode="annual">2023–26 · Jahresfamilien</button>
 </div><div id="researchChart" style="height:460px;width:100%"></div>
 <p id="researchChartNote" class="muted"></p><div class="muted researchCaveat">Separater Analysebericht: Das 2026er Dreierdepot hat 1.000 € insgesamt (333,33 € + 333,33 € + 333,34 €). Screenshot-Presets allein verwenden jeweils 1.000 € und sind daher nicht linear mit den Teildepots vergleichbar. 2026-Einstellungen wurden rückblickend betrachtet; keine unabhängige Validierung.</div>
 </div></details></div>
<aside class="right">'''
src=src.replace(needle,charts)
# stylesheet add at final style closing before body
needle='</style></head><body>'
assert src.count(needle)==1
css='''
.researchPanel{margin:14px 0 0;border:1px solid #385d75;background:linear-gradient(120deg,#0b1d2e,#102b3e);border-radius:14px;overflow:hidden;box-shadow:0 6px 20px #0003}
.researchPanel summary{padding:16px 19px;cursor:pointer;font-size:17px;font-weight:750;color:#c6f1eb}
.researchInner{padding:6px 17px 17px}
.researchControls{display:flex;flex-wrap:wrap;gap:8px;margin:5px 0 12px}
.researchControls button{font-size:12px;padding:9px 12px;background:#17324a;border:1px solid #34566e;color:#d6e9f5}
.researchControls button.active{background:#155e63;border-color:#76d2c4;color:#ecfffd}
.researchCaveat{border-top:1px solid #315168;padding-top:11px}
.highlightCharts,.applySaved{background:#1a5361;border-color:#6cc8c2;color:#eafff8;font-weight:700}
.savedPresets{border-color:#69aca8;background:linear-gradient(155deg,#0d2737,#0c2134)}
.savedNote{border:1px solid #385769;background:#112d41;color:#c0dbec;border-radius:8px;font-size:11.5px;line-height:1.6;padding:9px;margin:10px 0}
.savedNote strong{color:#f1e4b5}
#activeSnapshot{margin-top:6px;color:#76dcc4}
@media(max-width:850px){#researchChart{height:350px!important}.researchPanel summary{font-size:14px}}
'''
src=src.replace(needle,css+needle)
# inject chart data and replace app JS from v12 exact
chart=json.loads((OUT/'chart_data.json').read_text('utf-8'))
embedded='const EVIDENCE_CHARTS='+json.dumps(chart,separators=(',',':'),ensure_ascii=False)+';\n'
assert src.count((ROOT/'imba_genome_app_v12.js').read_text())==1
src=src.replace((ROOT/'imba_genome_app_v12.js').read_text(),embedded+app)
html_path=ROOT/'imba_genome_parameter_lab_v13_presets_charts.html'
html_path.write_text(src,encoding='utf-8')
print('BUILT',html_path,'bytes',html_path.stat().st_size)
print('app',OUT/'imba_genome_app_v13.js')