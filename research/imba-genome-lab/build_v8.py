from pathlib import Path
import base64, json, zipfile, numpy as np, pandas as pd
from plotly.offline.offline import get_plotlyjs
P=Path('/mnt/data'); W=P/'imba_genome_v8';V7=P/'imba_genome_v7'
z=np.load(V7/'full_4158_genomes_2026_daily.npz',allow_pickle=True)
ze=np.load(W/'extra_3168_mode_curves_2025_2026.npz',allow_pickle=True)
keys=['none','tp6','tp4','tp3','tp6_ttp_3_1p5','tp2','tp4_ttp_2_1','ttp_3_1p5','ttp_2_1']
def metax(name,s,b,key,mode,has2022=False):
    tp_on=key.startswith('tp');ttp_on=('ttp_' in key)
    tp=6 if key.startswith('tp6') else 4 if key.startswith('tp4') else 3 if key.startswith('tp3') else 2 if key.startswith('tp2') else 0
    act=3 if 'ttp_3_1p5' in key else 2 if 'ttp_2_1' in key else 0
    trail=1.5 if 'ttp_3_1p5' in key else 1 if 'ttp_2_1' in key else 0
    fam='both' if tp_on and ttp_on else 'tp' if tp_on else 'ttp' if ttp_on else 'none'
    return {'id':str(name),'s':int(s),'b':round(float(b),4),'priceMode':str(mode), 'exitKey':str(key),'exit':keys.index(key),
      'tp':tp_on,'ttp':ttp_on,'tpValue':tp,'act':act,'trail':trail,'family':fam,'has2022':has2022}
meta=[metax(*q,'renko_usd',True) for q in zip(z['ids'],z['sensitivity'],z['renko_box_usd'],z['exit_key'])]
meta.extend([metax(name,s,b,key,mode,False) for name,s,b,key,mode in zip(ze['ids'],ze['sens'],ze['param'],ze['exit_key'],ze['mode'])])
assert len(meta)==7326 and len({m['id'] for m in meta})==7326
u=pd.read_csv(P/'imba_fee006/imba_2022_pool_100_tp_ttp_fee006.csv')
byid={m['id']:i for i,m in enumerate(meta)}
raw=[byid[x] for x in u.loc[u.pool=='top100_raw','strategy_id'] if x in byid]
bal=[byid[x] for x in u.loc[u.pool=='top100_balanced','strategy_id'] if x in byid]

def factors(e,e25):
    e=e.astype(np.float64);e25=e25.astype(np.float64)
    f=np.empty_like(e);f25=np.empty_like(e25)
    f[:,0]=e[:,0];f[:,1:]=e[:,1:]/np.maximum(e[:,:-1],1e-12)
    f25[:,0]=e25[:,0];f25[:,1:]=e25[:,1:]/np.maximum(e25[:,:-1],1e-12)
    return np.concatenate([f25[:,-90:],f],axis=1).astype('<f4')
F=np.concatenate([factors(z['equity'],z['equity_2025']), factors(ze['equity'],ze['equity_2025'])],axis=0)
p22ref=pd.read_csv(P/'imba_fee006/imba_2022_tp_ttp_full_sweep_fee006.csv').set_index('strategy_id')
p22=np.array([float(p22ref.at[m['id'],'return_2022_1x_pct']) if m['has2022'] else np.nan for m in meta], dtype='<f4')
assert F.shape==(7326,398) and np.isfinite(F).all() and (F>0).all()

html_src=(V7/'build_v7.py').read_text()
html=html_src.split("html=r'''",1)[1].split("'''",1)[0]
html=html.replace('v7 · 4.158 Backtests','v8 · 7.326 Backtests').replace('v7</span>','v8</span>')
html=html.replace('4.158 tatsächlich gerechnete Bot-Genkombinationen','7.326 tatsächlich gerechnete Bot-Genkombinationen')
html=html.replace('Alle 4.158','Alle im Marktmodus')
html=html.replace('Alle 4.158 · Exit × Sens','Alle im Marktmodus · Exit × Sens')
html=html.replace('Der neue 4.158er-Sweep','Der erweiterte 7.326er-Sweep')
html=html.replace('Keine erfundenen Rasterpunkte','USD-Renko / %-Renko / Candles · nur echte Backtests')
# Make mode controls first-class, positioned above chart controls.
mode_html='''<section class="bar" id="modeBar" style="border-color:#3b7881;align-items:center;background:linear-gradient(105deg,#10273d,#102d37)">
 <div class="item" style="min-width:210px"><span class="smallLabel">PREIS-BASIS / STRATEGIE-GEN</span><strong style="font-size:15px;color:#84e2cf">Candles ↔ Renko</strong></div>
 <label class="toggle"><input type="checkbox" id="useCandles"> Normale 5m-Candles</label>
 <label class="toggle"><input type="checkbox" id="usePercent"> Renko prozentual (%)</label>
 <span class="pill" id="modeTag">Renko · fester USD-Abstand</span>
 <label class="toggle"><input type="checkbox" id="useAllBoxes" checked> Alle getesteten Boxgrößen</label>
 <div class="modeSlider" style="flex:1;min-width:240px"><span class="smallLabel">BOXGRÖSSE · 0,1ER-SCHRITTE</span><div style="display:flex;gap:11px;align-items:center"><input id="boxStep" type="range" class="range" min="1" max="10" step="1" value="3" style="flex:1"><strong id="boxStepLabel" style="min-width:74px;text-align:right">0,30 USD</strong></div></div>
 <span class="muted" id="modeExplanation" style="max-width:295px">Alle angezeigten Kombinationen sind tatsächlich getestet.</span>
</section>'''
html=html.replace('<section class="bar">\n <div class="item"><label>PERFORMANCEZEITRAUM',mode_html+'\n<section class="bar">\n <div class="item"><label>PERFORMANCEZEITRAUM')
html=html.replace('.range{accent-color:#55cdd5}', '.range{accent-color:#55cdd5}.toggle{display:inline-flex;gap:7px;align-items:center;padding:9px 11px;cursor:pointer;border:1px solid #396176;background:#10283d;border-radius:9px;white-space:nowrap}.toggle input{accent-color:#58cdb6;width:17px;height:17px}.toggle:has(input:checked){background:#153f48;border-color:#5dcbb5;color:#91f5dc}#boxStep:disabled{opacity:.25}')
html=html.replace('RENKO-FILTER','BOX-FILTER (GENAU)')
html=html.replace('Die Zahl am Gen zeigt die Zahl betroffener Strategien, nicht die Tradeanzahl.', 'Die Zahl am Gen zeigt, wie viele ausgewählte Strategien es verwenden. Candle- und Renko-Gen zählen getrennt.')
html=html.replace('bei nur vier TP-Werten', 'bei nur vier TP-Werten')
html=html.replace('Alle eingeschlossenen Strategien nehmen an der Top-6-Rotation teil.', 'Nur wirklich berechnete Kombinationen gelangen in die Top-6-Auswahl.')
html=html.replace('2022 · volles Jahr','2022 · Archiv: nur USD-Renko')
html=html.replace('<div class="muted" style="margin-top:7px">Die Zahl', '<div class="muted" style="margin-top:7px">Die Zahl')

js=(V7/'app_v7.js').read_text()
js=js.replace('Quant Genome v7','Quant Genome v8')
js=js.replace("['renko','▦','RENKO','#70a8ed']", "['renko','▦','RENKO','#70a8ed'],['candle','▤','CANDLES','#78d3ef']")
js=js.replace("let state={day:DAYS-1", "let state={priceMode:'renko_usd',day:DAYS-1")
js=js.replace("function value(i){return state.dataset==='2022'?P22[i]:rPct(i,state.day,state.horizon);}",
              "function value(i){return state.dataset==='2022'?(META[i].has2022?P22[i]:NaN):rPct(i,state.day,state.horizon);}")
js=js.replace("if(state.mode==='manual')return [...state.manual];\n return ALL;", "if(state.mode==='manual')return [...state.manual];\n return ALL;")
js=js.replace(" let ids=getPresetIds().filter(i=>state.family==='all'||META[i].family===state.family);",
              " let ids=getPresetIds().filter(i=>META[i].priceMode===state.priceMode && (state.dataset!=='2022'||META[i].has2022) && (state.family==='all'||META[i].family===state.family));")
js=js.replace(" if(state.box!=='all')ids=ids.filter(i=>Math.abs(META[i].b-Number(state.box))<.00001);",
              " if(state.box!=='all' && state.priceMode!=='candle')ids=ids.filter(i=>Math.abs(META[i].b-Number(state.box))<.00001);")
js=js.replace(" populateSelector('boxFilter',[['all','Alle Boxgrößen (Aggregation)'],...BOXES.map(b=>[b.toFixed(2),b.toFixed(2)+' USD (exakt)'])],'all');",
              " populateSelector('boxFilter',[['all','Alle Boxgrößen (Aggregation)'],...BOXES.map(b=>[b.toFixed(2),b.toFixed(2)+' USD (exakt)'])],'all');")
# Correct fallback if clicked single-cell markers
js=js.replace("function value(i){return state.dataset==='2022'?(META[i].has2022?P22[i]:NaN):rPct(i,state.day,state.horizon);}",
              "function value(i){return state.dataset==='2022'?(META[i].has2022?P22[i]:NaN):rPct(i,state.day,state.horizon);}")
# Gene label and count dynamic
js=js.replace(" const count=list.filter(i=>key==='imba'||key==='renko'||META[i][key]).length;",
              " const count=list.filter(i=>key==='imba'||(key==='renko'&&META[i].priceMode!=='candle')||(key==='candle'&&META[i].priceMode==='candle')||META[i][key]===true).length;")
js=js.replace(" const m=META[id],d=choosePortfolio()[state.day];const isLive=d.live.includes(id),chosen=d.ids.includes(id);",
              " const m=META[id],d=choosePortfolio()[state.day];const isLive=d.live.includes(id),chosen=d.ids.includes(id);")
js=js.replace(" $('inspectGenes').innerHTML=`<span>⌁ IMBA S${m.s}</span><span>▦ Renko ${number(m.b,2)} USD</span>`+",
              " $('inspectGenes').innerHTML=`<span>⌁ IMBA S${m.s}</span><span>${m.priceMode==='candle'?'▤ Candles 5m':m.priceMode==='renko_pct'?'▦ Renko '+number(m.b,1)+' %':'▦ Renko '+number(m.b,2)+' USD'}</span>`+")
js=js.replace("function layoutAxes(){const axis=k=>({title:{text:AXES[k],font:{size:12}}",
              "function layoutAxes(){const axis=k=>({title:{text:k==='b'?(state.priceMode==='renko_pct'?'Renko-Box (%)':state.priceMode==='candle'?'Candles · ohne Box':'Renko-Box (USD)'):AXES[k],font:{size:12}}")
js=js.replace("state.x='s';state.y='b';state.family='none';state.box='all';state.mode='original462';",
              "state.x='s';state.y='b';state.family='none';state.box='all';state.priceMode='renko_usd';state.mode='original462';")
js=js.replace(" setControls();changePool();\n}\nfunction serialize()", " syncModeControls();setControls();changePool();\n}\nfunction serialize()")
js=js.replace("function serialize(){return {version:7,date:DATES[state.day]", "function serialize(){return {version:8,priceMode:state.priceMode,date:DATES[state.day]")
js=js.replace("a.download='quant_genome_v7_preset.json'", "a.download='quant_genome_v8_preset.json'")
js=js.replace("if(x.version!==7)throw Error('Bitte eine v7 Preset-Datei wählen.');", "if(![7,8].includes(x.version))throw Error('Bitte eine v7/v8 Preset-Datei wählen.');")
js=js.replace("   state.family=['all','none','tp','ttp','both'].includes(x.family)?x.family:'tp';", "   state.family=['all','none','tp','ttp','both'].includes(x.family)?x.family:'tp';\n   state.priceMode=['candle','renko_pct','renko_usd'].includes(x.priceMode)?x.priceMode:'renko_usd';")
js=js.replace("   setControls();$('horizon').value=state.horizon;", "   syncModeControls();setControls();$('horizon').value=state.horizon;")
js=js.replace("a.download='quant_genome_v7_echte_kandidaten.csv'", "a.download='quant_genome_v8_echte_kandidaten.csv'")
js=js.replace("const lines=['strategy_id,sensitivity,renko_box_usd,exit_program,tp_pct", "const lines=['strategy_id,price_mode,sensitivity,renko_box_value,exit_program,tp_pct")
js=js.replace("lines.push([m.id,m.s,m.b,m.exitKey", "lines.push([m.id,m.priceMode,m.s,m.b,m.exitKey")
js=js.replace(" $('plotCountNote').textContent=`${len} echte Bot-Kurven · ${state.grid.occupied} belegte Parameterzellen · ${state.box==='all'?'Boxgrößen aggregiert':'Box '+state.box} · 10× nur im Portfolio`;",
              " $('plotCountNote').textContent=`${len} echte Bot-Kurven · ${state.grid.occupied} belegte Parameterzellen · ${state.priceMode==='candle'?'5-Minuten-Candles':state.priceMode==='renko_pct'?'Prozent-Renko':'USD-Renko'} · ${state.box==='all'?'Alle getesteten Größen': 'Box '+state.box} · 10× nur im Portfolio`;")
js=js.replace(" $('dataWarning').textContent=state.dataset==='2022'?'2022: Archivierter vollständiger 1×-Jahressweep, nicht 2026er Handelsgewinn. Portfolio bleibt 2026.':'2026: Neu berechneter 5m-Close-Forschungs-Backtest. Nicht identisch mit dem archivierten v6-Portfolio.';",
              " $('dataWarning').textContent=state.dataset==='2022'?'2022: Nur ursprünglicher USD-Renko-Sweep verfügbar. Prozent-Renko und Candles wurden nur 2025/2026 neu berechnet. Portfolio bleibt 2026.':'2026: Alle sichtbaren Bot-Kurven wurden für den jeweils gewählten Marktmodus berechnet (5m, Forschungsmodell; keine TradingView-Parität).';")
# avoid duplicate `NaN`/emptiness and infinite zrange errors in empty mode: fallback to 2D with no trace when no entries; existing maybe 
js=js.replace(" const bg='#102034',zrange=[Math.min(-50,...g.z.flat().filter(Number.isFinite)),Math.max(80,...g.z.flat().filter(Number.isFinite))];",
              " const vals=g.z.flat().filter(Number.isFinite); const bg='#102034',zrange=[Math.min(-50,...vals),Math.max(80,...vals)];")
# switch mode functions placed before UI binding
insertion="""
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
"""
js=js.replace('function bindUI(){',insertion+'\nfunction bindUI(){')
js=js.replace(" initOptions();\n if(!supportsWebGL())", " initOptions();\n syncModeControls();\n $('useCandles').onchange=e=>switchPriceMode(e.target.checked?'candle':'renko_usd');\n $('usePercent').onchange=e=>switchPriceMode(e.target.checked?'renko_pct':'renko_usd');\n $('useAllBoxes').onchange=e=>{state.box=e.target.checked?'all':(Number($('boxStep').value)/10).toFixed(2);syncModeControls();changePool();};\n $('boxStep').oninput=e=>{state.box=(Number(e.target.value)/10).toFixed(2);$('boxStepLabel').textContent=(Number(e.target.value)/10).toFixed(1)+(state.priceMode==='renko_pct'?' %':' USD');};\n $('boxStep').onchange=()=>{state.box=(Number($('boxStep').value)/10).toFixed(2);$('useAllBoxes').checked=false;syncModeControls();changePool();};\n if(!supportsWebGL())")
# Need state.box change via original boxFilter handler update slider and all box status
js=js.replace("$(el).onchange=e=>{state[field]=e.target.value;if((field==='x'||field==='y')", "$(el).onchange=e=>{state[field]=e.target.value;if(field==='box'){if(state.box!=='all')$('boxStep').value=Math.round(Number(state.box)*10);syncModeControls();}if((field==='x'||field==='y')")
# 2022 no new data: filter enforced + route dataset in 2022 mode USD. 
js=js.replace(" $('dataset').onchange=e=>{state.dataset=e.target.value;state.grid=buildHeatmap();setStatus();renderChart();};",
              " $('dataset').onchange=e=>{state.dataset=e.target.value;if(state.dataset==='2022'&&state.priceMode!=='renko_usd')switchPriceMode('renko_usd');else changePool();};")
js=js.replace("  const group=state.grid.groups.get(Number(pt.x)+'|'+Number(pt.y));", "  const group=state.grid.groups.get(Number(pt.x)+'|'+Number(pt.y));")
# Guard chart errors when top6 uses few positive, same original

# Make 'all' preset keep current market mode, fixed preset forces USD.
# JS modules replacing old summary to new state

subs={
 '%%PLOTLY%%':get_plotlyjs(),
 '%%META%%':json.dumps(meta,separators=(',',':'),ensure_ascii=False),
 '%%DATES%%':json.dumps(pd.date_range('2026-01-01','2026-11-04',freq='D').strftime('%Y-%m-%d').tolist()),
 '%%BOXES%%':json.dumps(sorted({m['b'] for m in meta if m['priceMode']=='renko_usd'})),
 '%%ORIGINAL%%':json.dumps([i for i,m in enumerate(meta) if m['family']=='none' and m['priceMode']=='renko_usd' and m['b']<=.50]),
 '%%ALL%%':json.dumps(list(range(len(meta)))),
 '%%RAW100%%':json.dumps(raw),
 '%%BAL100%%':json.dumps(bal),
 '%%F_B64%%':json.dumps(base64.b64encode(F.tobytes()).decode()),
 '%%P22_B64%%':json.dumps(base64.b64encode(p22.tobytes()).decode()),
 '%%APPJS%%':js,
}
for k,v in subs.items():
 if k not in html:raise KeyError(k)
 html=html.replace(k,v)
path=P/'imba_2026_genome_parameter_lab_v8_renko_candles.html'
path.write_text(html,encoding='utf-8')
(W/'app_v8.js').write_text(js)
(W/'README_V8.md').write_text(f'''# Quant Genome Parameter Lab v8

- Alle {len(meta):,} Strategie-Gen-Kombinationen haben eigene 2025 und 2026 Backtest-Kurven.
- Ursprüngliche 4158 Strategien (feste Renko-Boxen USD 0,10–0,50, Schritt 0,02).
- Zusätzlich 990 neue feste USD-Renko-Varianten mit Box 0,6–1,0 USD (Schritt 0,1).
- Zusätzlich 1980 prozentuale Renko-Varianten mit dynamischer Box 0,1–1,0 % des aktuellen Renko-Ankerpreises (Schritt 0,1 Prozentpunkte); 2-Box-Umkehr.
- Zusätzlich 198 IMBA-Varianten auf echten 5-Minuten-OHLC-Candles ohne Renko; Sensitivity × 10 Kerzen für Grenzen.
- Jede Eingabe enthält neun TP-/TTP-Exitprogramme und 22 Sensitivities.
- Renko-Modi: Nur getestete Schritte werden angeboten, Slider für Boxgröße, All-Boxes-Auswahl oder exakter Filter. Candle-Modus setzt Renko-Box-Gen aus.
- 0,06 % Gebühren je Orderseite, nächster 5-Minuten-Schluss-Fill, 1× Tagesrenditen; 10× Portfolio-Tagesmodell, keine TV-Parität.
- 2022 nur mit 4158 historischen USD-Renko-Varianten (die vorliegenden Tagesreihen 2025/26 sind für neue Preismodi; kein 2022er Nachlauf erfunden).
- In 2026 Top-6-Walk-forward nur positiv vergangene 14-Tage-Rendite; individueller Stopp.
- Als neue Prozent-Renko-Basis wird pro Schritt die Box dynamisch vom letzten Renko-Anker berechnet. Diese Variante ist eine explizite Forschungsannahme und keine bestätigte TradingView-Darstellungsparität.
- WICHTIG: Für Candles ist IMBA analog mit 5-Minuten-OHLC-High/Low-Kanal (S×10 Candle Lookback), nicht identisch zur Renko-Brick-Definition. Unterschiedliche Feature-Geometrie wird bewusst verglichen.

Keine Live-Bot-Änderungen. Die neue kombinierte v8-Datenbasis ist im ZIP enthalten.
''')
import zipfile as zp
zpath=P/'imba_2026_genome_parameter_lab_v8_offline.zip'
with zp.ZipFile(zpath,'w',compression=zp.ZIP_DEFLATED,compresslevel=6) as zz:
 zz.write(path,arcname=path.name)
 for name in ['app_v8.js','build_new_modes.py','extra_3168_mode_curves_2025_2026.npz','extra_3168_annual_2025_2026.csv','README_V8.md']:
  zz.write(W/name,arcname='research/'+name)
 zz.write(P/'imba_genome_v7/full_4158_genomes_2026_daily.npz',arcname='data/full_4158_genomes_2026_daily.npz')
 zz.write(P/'imba_genome_v7/build_v7.py',arcname='research/build_v7.py')
 zz.write(P/'imba_fee006/imba_2022_tp_ttp_full_sweep_fee006.csv',arcname='data/imba_2022_tp_ttp_full_sweep_fee006.csv')
print('BUILD',path,'size',round(path.stat().st_size/1e6,1),'MB zip',round(zpath.stat().st_size/1e6,1),'MB')