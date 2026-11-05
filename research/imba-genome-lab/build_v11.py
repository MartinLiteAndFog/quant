"""Add selectable 7- and 10-day Top-6 ranking windows to existing v10 HTML.
Only selection lookback changes; Top-6 rotation remains every 14 days.
Requires existing imba_genome_parameter_lab_v10_ui.html and imba_genome_app_v10.js.
"""
from pathlib import Path

P=Path(__file__).resolve().parent
html=P.joinpath('imba_genome_parameter_lab_v10_ui.html').read_text(encoding='utf-8')
old='<select id="ranking"><option value="14" selected>Vorherige 14 Tage</option><option value="30">Vorherige 30 Tage</option><option value="90">Vorherige 90 Tage</option></select>'
new='<select id="ranking"><option value="7">Letzte 7 Tage (1 Woche)</option><option value="10">Letzte 10 Tage</option><option value="14" selected>Letzte 14 Tage</option><option value="30">Letzte 30 Tage</option><option value="90">Letzte 90 Tage</option></select>'
assert html.count(old)==1
html=html.replace(old,new)
html=html.replace('Genome Parameter Lab v10','Genome Parameter Lab v11').replace('Parameter Lab v10','Parameter Lab v11').replace('>v10</span>','>v11</span>')
html=html.replace('RANKING AUS VERGANGENHEIT','TOP-6-RANKING · RÜCKBLICK')
html=html.replace('Nur positive Vorperiodenrenditen · höchstens 6 aktive Bots · 14-tägige Neuwahl',
                  'Nur positive Renditen im gewählten Rückblick · höchstens 6 aktive Bots · weiterhin alle 14 Tage Neuwahl')
html=html.replace('version:10,priceMode','version:11,priceMode')
html=html.replace('if(![7,8,9,10].includes(x.version))','if(![7,8,9,10,11].includes(x.version))')
html=html.replace('quant_genome_v10_preset.json','quant_genome_v11_preset.json')
P.joinpath('imba_genome_parameter_lab_v11_ranking_7_10_tage.html').write_text(html,encoding='utf-8')

js=P.joinpath('imba_genome_app_v10.js').read_text(encoding='utf-8')
js=js.replace('Quant Genome v10','Quant Genome v11')
js=js.replace('version:10,priceMode','version:11,priceMode')
js=js.replace('if(![7,8,9,10].includes(x.version))','if(![7,8,9,10,11].includes(x.version))')
js=js.replace('quant_genome_v10_preset.json','quant_genome_v11_preset.json')
assert 'score(i,day){return rPct(i,day-1,state.rank);}' in js
P.joinpath('imba_genome_app_v11.js').write_text(js,encoding='utf-8')