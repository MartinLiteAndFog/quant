"""Patch research v9 to a more fluid, collapsible v10 UI. Backtest ledger stays unchanged.

Input: imba_genome_parameter_lab_v9_2022_2026.html, imba_genome_v9/app_v9.js.
Output: imba_genome_parameter_lab_v10_ui.html, imba_genome_v10/app_v10.js.
"""
from pathlib import Path
import re
P=Path('/mnt/data'); W=P/'imba_genome_v10';W.mkdir(exist_ok=True)
old=(P/'imba_genome_parameter_lab_v9_2022_2026.html').read_text()
js=(P/'imba_genome_v9/app_v9.js').read_text()
assert 'function makePlotData()' in js and 'function bindUI()' in js
js=js.replace('/* Quant Genome v8', '/* Quant Genome v10')
# Motion/smoothing affect only presentation, not stored 1x curves or trades
js=js.replace("surfaceMode:'overlay',layers:", "smooth:1,playMs:500,sidebarOpen:true,surfaceMode:'overlay',layers:")
# visually smooth mesh, but preserve all untested cells and unmodified hover text/portfolio
before='function makePlotData()'
helper='''// Display-only smoothing: 3x3 weighted convolution on existing cells;
// original backtest data, hover text, portfolio ledger and selection remain unchanged.
function softenedMesh(matrix,passes){
 if(!passes)return matrix;
 let z=matrix.map(row=>row.slice());
 for(let pass=0;pass<passes;pass++){
  const src=z;z=src.map((row,y)=>row.map((v,x)=>{
   if(v===null||!Number.isFinite(v))return null;
   let sum=4*v,weight=4;
   for(const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]){
    const w=(dx===0||dy===0)?2:1;
    const neighbor=src[y+dy]?.[x+dx];
    if(neighbor!==null && Number.isFinite(neighbor)){sum+=w*neighbor;weight+=w;}
   }
   return sum/weight;
  }));
 }
 return z;
}
function surfaceStyle(){return {lighting:{ambient:.85,diffuse:.72,roughness:.91,specular:.09,fresnel:.07},lightposition:{x:500,y:-250,z:900},hidesurface:false};}
'''
js=js.replace(before,helper+before)
js=js.replace("x:layer.xs,y:layer.ys,z:layer.z,", "x:layer.xs,y:layer.ys,z:softenedMesh(layer.z,state.smooth),...surfaceStyle(),")
js=js.replace("x:g.xs,y:g.ys,z:g.z,colorscale:scale,opacity:.90", "x:g.xs,y:g.ys,z:softenedMesh(g.z,state.smooth),...surfaceStyle(),colorscale:scale,opacity:.90")
# Keep 2D heatmap exact (no aesthetic smoothing).
# preserve interactive camera through uirevision when time slider moves.
js=js.replace("uirevision:'camera'", "uirevision:'camera',transition:{duration:Math.min(state.playMs,240),easing:'cubic-in-out'}")
# rAF pacing, coalesces frame updates while plotly draws; can't claim native true 60Hz 3D
oldblock=""" function play(dir){stop();state.dir=dir;state.playing=setInterval(()=>{const nx=state.day+dir;if(nx<0||nx>=DAYS){stop();return;}setDay(nx);},Math.max(100,Number($('speed').value)));}"""
newblock=""" function play(dir){
  stop();state.dir=dir;
  // requestAnimationFrame coalesces updates; never queue up expensive WebGL plots.
  let elapsed=0,last=0;
  function tick(ts){
   if(state.playing===null)return;
   if(!last)last=ts;
   elapsed+=Math.max(0,ts-last);last=ts;
   if(elapsed>=state.playMs){
    elapsed%=state.playMs;
    const nx=state.day+dir;
    if(nx<0||nx>=DAYS){stop();return;}
    setDay(nx);
   }
   state.playing=requestAnimationFrame(tick);
  }
  state.playing=requestAnimationFrame(tick);
 }"""
assert oldblock in js
js=js.replace(oldblock,newblock)
js=js.replace("function stop(){if(state.playing!==null){clearInterval(state.playing);state.playing=null;}}",
              "function stop(){if(state.playing!==null){cancelAnimationFrame(state.playing);state.playing=null;}}")
js=js.replace("if(state.playing!==null){clearInterval(state.playing);state.playing=null;}","if(state.playing!==null){cancelAnimationFrame(state.playing);state.playing=null;}")
# wire new settings
marker=" $('surfaceMode').onchange=e=>{state.surfaceMode=e.target.value;renderChart();};"
assert marker in js
js=js.replace(marker,''' $('smoothLevel').onchange=e=>{state.smooth=Number(e.target.value);renderChart();};
 $('sidebarToggle').onclick=()=>{
  state.sidebarOpen=!state.sidebarOpen;
  document.querySelector('.cols').classList.toggle('sidebarHidden',!state.sidebarOpen);
  $('sidebarToggle').textContent=state.sidebarOpen?'Seitenmenüs ausblenden':'Seitenmenüs einblenden';
  setTimeout(()=>{Plotly.Plots.resize($('chart'));},100);
 };
 document.querySelectorAll('aside.right > section.panel').forEach((panel,i)=>{
  const title=panel.querySelector('h2');if(!title)return;
  title.setAttribute('role','button');title.setAttribute('tabindex','0');
  title.setAttribute('aria-expanded',i<3?'true':'false');
  panel.classList.toggle('folded',i>=3);
  const toggle=()=>{panel.classList.toggle('folded');title.setAttribute('aria-expanded',String(!panel.classList.contains('folded')));};
  title.addEventListener('click',toggle);
  title.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();toggle();}});
 });
 $('speed').onchange=e=>{state.playMs=Number(e.target.value);};
'''+marker)
# make sure controls reflect actual speed selected
js=js.replace("smooth:1,playMs:500", "smooth:1,playMs:650")
# fix version 9 preset import (bug observed)
js=js.replace("if(![7,8].includes(x.version))", "if(![7,8,9,10].includes(x.version))")
js=js.replace("version:9,priceMode", "version:10,priceMode")
js=js.replace("quant_genome_v9_preset.json", "quant_genome_v10_preset.json")
js=js.replace("if(x.surfaceMode)state.surfaceMode=x.surfaceMode;", "if(x.smooth!==undefined){state.smooth=Number(x.smooth);$('smoothLevel').value=String(state.smooth);}if(x.surfaceMode)state.surfaceMode=x.surfaceMode;")
js=js.replace("year:state.dataset,surfaceMode:state.surfaceMode", "year:state.dataset,smooth:state.smooth,surfaceMode:state.surfaceMode")
# page HTML changes; scripts with large JSON stay embedded/offline
html=old.replace('Parameter Lab v9','Parameter Lab v10').replace('Genome Parameter Lab v9','Genome Parameter Lab v10').replace('>v9</span>','>v10</span>')
html=html.replace('class="cols"','class="cols"')
html=html.replace('<div class="cols"><div>', '''<div class="workspaceSwitch"><button type="button" id="sidebarToggle">Seitenmenüs ausblenden</button><span class="smallLabel">3D-Flächen: Glättung ändert nur die Darstellung, nicht die Gewinne oder Top-6-Auswahl.</span></div><div class="cols"><div>''')
html=html.replace('<section class="bar timebar">', '''<section class="bar timebar"><div class="item"><label>3D-GLÄTTUNG · NUR OPTISCH</label><select id="smoothLevel"><option value="0">Aus · Rohwerte</option><option value="1" selected>Sanft</option><option value="2">Weicher</option></select></div>''')
html=html.replace('<option value="250">Schnell</option><option value="650" selected>Mittel</option><option value="1300">Langsam</option>', '<option value="340">Schnell</option><option value="650" selected>Mittel</option><option value="1150">Langsam</option>')
# Visuals: subtle smoother edges & collapsible sidebar
css='''
.workspaceSwitch{display:flex;justify-content:flex-end;align-items:center;gap:12px;margin:-3px 0 8px}
.chartpanel{border-radius:17px;background:radial-gradient(ellipse at 60% 35%,#173a4b 0,#102034 58%,#0a1a2c 100%);box-shadow:0 12px 30px #0003,inset 0 0 0 1px #4369803c}
.panel{border-radius:13px;transition:border-color 150ms ease,background 150ms ease}
aside.right .panel>h2{cursor:pointer;position:relative;padding:4px 22px 5px 0;user-select:none}
aside.right .panel>h2::after{content:'−';position:absolute;right:3px;color:#69cbbd;font-size:18px;top:0}
aside.right .panel.folded>h2::after{content:'+'}
aside.right .panel.folded>:not(h2){display:none}
aside.right .panel.folded{background:#0c1c2d}
.cols.sidebarHidden{grid-template-columns:minmax(0,1fr)}
.cols.sidebarHidden>aside.right{display:none}
#sidebarToggle{background:#133f4b;border-color:#5293a0;color:#c2efe8}
.bar,.panel,.hud{box-shadow:0 3px 16px #00000016}
#chart{transition:filter .24s ease}
'''
html=html.replace('</style>', css+'</style>',1)
# robust: exactly replace app script, not plotly bundled
start=html.index('function decodeF32(b64)') # search just to ensure in final
app_start=html.index('const $=id=>document.getElementById(id);')
app_end=html.index('</script></body></html>',app_start)
html=html[:app_start]+js+html[app_end:]
assert 'Genome Parameter Lab v10' in html and 'sidebarToggle' in html and 'softenedMesh' in html
out=P/'imba_genome_parameter_lab_v10_ui.html';out.write_text(html)
(W/'app_v10.js').write_text(js)
print('wrote',out,'sizeMB',round(out.stat().st_size/1024**2,2),'JS lines',len(js.splitlines()))