"""Data-backed charts and inspectable tables for Genome Lab v13.
Chart #1 three actual independently managed 333.33/333.33/333.34 sleeves.
Chart #2 Sharpe vs MDD and normalized benchmarks (seed 1000 each).
Chart #3 Chained prior-year Top100 Raw annual rotation (different experiment).
"""
from pathlib import Path
import sys
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
P=Path('/mnt/data/imba_genome_v13_work')
root=Path('/mnt/data');annual=root/'imba_annual_raw100_2022_2026'
colors={'original462':'#61bdd8','raw100':'#f3ad71','balanced100':'#c3a1f4','mixed':'#e7d27a','historical':'#8697ab','rotate':'#7dcdb4','accum':'#c9aaec'}
plt.rcParams.update({'figure.facecolor':'#081728','axes.facecolor':'#0d2536','savefig.facecolor':'#081728',
  'axes.edgecolor':'#496379','axes.labelcolor':'#bdd2e0','text.color':'#e6f0fa','xtick.color':'#9db7c9','ytick.color':'#9db7c9',
  'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':16,'axes.titleweight':'bold','grid.color':'#375065',
  'grid.alpha':.37,'legend.facecolor':'#102a3d','legend.edgecolor':'#4f6677','legend.labelcolor':'#dceaf2'})

def save(fig,name):
 fig.savefig(P/(name+'.png'),dpi=155,bbox_inches='tight',pad_inches=.22)
 fig.savefig(P/(name+'.svg'),bbox_inches='tight',pad_inches=.22)
 plt.close(fig)

mix=pd.read_csv(P/'screenshot_mix_2026_daily.csv',parse_dates=['date'])
base=pd.read_csv(root/'imba_mix_2026/portfolio_mix_2026_daily.csv',parse_dates=['date'])
assert np.all(mix.date.values==base.date.values)
fig,ax=plt.subplots(figsize=(12.3,5.6),constrained_layout=True)
for key,label in [('original462','Original 462 · 333,33 €'),('raw100','Top100 Raw · 333,33 €'),('balanced100','Top100 Balanced B · 333,34 €')]:
 ax.plot(mix.date,mix[key],label=label,color=colors[key],linewidth=1.65,alpha=.85)
ax.plot(mix.date,mix.mixed,color=colors['mixed'],linewidth=2.9,label=f'Drei-Büschel-Mix: {mix.mixed.iloc[-1]:,.0f} €')
ax.plot(base.date,base.mix,color='#98a5b7',linewidth=1.1,alpha=.7,linestyle=':',label='Voriger Mix: Balanced −30 %')
ax.axhline(1000,color='#aec5d4',linewidth=.8,linestyle='--',alpha=.65)
ax.grid(True,axis='y');ax.set_ylabel('Depotwert (€)');ax.set_xlabel('2026')
ax.set_title('2026 · Drei unabhängige Büschel, 1.000 € Gesamtstart')
ax.text(.01,.94,'10×-Tagesmodell · keine externen Einzahlungen · bis 04.11.2026',transform=ax.transAxes,color='#a7b9cb',fontsize=10,va='top')
ax.legend(loc='upper left',bbox_to_anchor=(.01,.85),fontsize=9)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:,.0f}'.replace(',','.')))
save(fig,'2026_drei_bueschel_mix_kapitalkurve')

standalone={
 'Original 462':('shot_b_original14_stop20',1000,'#61bdd8'),
 'Top100 Raw':('raw100_comparator_30_stop30',1000,'#f3ad71'),
 'Top100 Balanced B':('shot_c_balanced30_stop25',1000,'#c3a1f4'),
 'Drei-Büschel-Mix':(None,1000,'#e7d27a'),
}
stats=[]
for label,(key,seed,c) in standalone.items():
 if key is not None:series=pd.read_csv(P/(key+'_1000.csv')).wealth.to_numpy()
 else:series=mix.mixed.to_numpy()
 ret=series/np.r_[seed,series[:-1]]-1
 sh=float(ret.mean()/ret.std(ddof=1)*np.sqrt(365))
 dd=float(np.min(series/np.maximum.accumulate(np.r_[seed,series])[1:]-1)*100)
 stats.append({'portfolio':label,'seed':seed,'end_eur':series[-1],'sharpe_365d':sh,'max_drawdown_pct':dd,'return_pct':(series[-1]/seed-1)*100})
statsdf=pd.DataFrame(stats);statsdf.to_csv(P/'screenshot_mix_2026_sharpe.csv',index=False)
fig,(a,b)=plt.subplots(1,2,figsize=(12.3,4.6),constrained_layout=True)
labels=statsdf.portfolio.tolist()
cs=[standalone[l][2] for l in labels]
x=np.arange(len(labels))
a.bar(x,statsdf.sharpe_365d,color=cs,width=.65)
a.set_xticks(x,labels,rotation=15,ha='right');a.set_ylabel('Sharpe (365 Tagesrenditen p.a.)');a.set_title('Sharpe Ratio · 2026')
b.bar(x,statsdf.max_drawdown_pct,color=cs,width=.65)
b.set_xticks(x,labels,rotation=15,ha='right');b.set_ylabel('Max. Drawdown (%)');b.set_title('Maximaler Einbruch · 2026')
for v in [a,b]:v.grid(axis='y');v.set_axisbelow(True)
for xi,v in zip(x,statsdf.sharpe_365d):a.text(xi,v+.07,f'{v:.2f}',ha='center',fontsize=10)
for xi,v in zip(x,statsdf.max_drawdown_pct):b.text(xi,v-3,f'{v:.1f} %',ha='center',fontsize=10)
save(fig,'2026_sharpe_drawdown_vergleich')

annualdf=pd.read_csv(annual/'browser_verified_chained_daily.csv',parse_dates=['date'])
annualdf=annualdf[annualdf.score_days==14]
annuals={'carry_2022':('2022er Top100 behalten','#f3ad71'),
 'annual_rotate':('Top100 jährlich ersetzen','#70d6ab'),
 'accumulate_past':('Jahrgänge akkumulieren','#c5a7ed')}
fig,ax=plt.subplots(figsize=(12.3,5.2),constrained_layout=True)
for code,(lab,color) in annuals.items():
 d=annualdf[annualdf.method==code].sort_values('date')
 ax.plot(d.date,np.maximum(d.wealth,.01),label=lab,color=color,linewidth=2.1)
ax.set_yscale('log');ax.axhline(1000,color='#bacddb',linestyle='--',linewidth=.8,alpha=.7)
ax.set_ylim(.01,3e4);ax.grid(axis='y',which='both');ax.set_ylabel('Depotwert (€), logarithmische Skala')
ax.set_xlabel('2023–2026');ax.set_title('Top-100-Jahresfamilien · Kapital durchgehend ab 2023')
ax.legend(loc='upper right',fontsize=10)
ax.text(.01,.03,'Andere Studie: jährliche Familienrotation, 14-Tage-Ranking, Einzelstopp −30 %, 10×-Tagesmodell.',transform=ax.transAxes,color='#a7b9cb',fontsize=10)
save(fig,'2023_2026_jahresfamilien_chained')

# prepare concise JSON chart bundle for app, ~4*308 + 3*1404 points
import json
payload={
 'mix':{'dates':mix.date.dt.strftime('%Y-%m-%d').tolist(),
        'original':mix.original462.round(2).tolist(),
        'raw':mix.raw100.round(2).tolist(),
        'balanced':mix.balanced100.round(2).tolist(),
        'mixed':mix.mixed.round(2).tolist(),
        'previous':base['mix'].round(2).tolist()},
 'stats':statsdf.round(4).to_dict('records'),
 'annual':{
 code:{'dates':annualdf.query('method == @code').sort_values('date').date.dt.strftime('%Y-%m-%d').tolist(),
       'wealth':annualdf.query('method == @code').sort_values('date').wealth.round(3).tolist(),
       'label':label}
 for code,(label,_) in annuals.items()},
 'annual_genes':pd.read_csv(annual/'annual_top100_genes_summary.csv').to_dict('records')
}
(P/'chart_data.json').write_text(json.dumps(payload,separators=(',',':')),encoding='utf8')
print(statsdf.to_string(index=False))
for fn in P.glob('*.png'):print('chart',fn.name,fn.stat().st_size)