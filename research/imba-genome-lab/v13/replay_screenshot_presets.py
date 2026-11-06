"""Exact Chromium v12 JS 2026 ledger from three screenshot presets.
No external deposits; 2026 daily factor snapshot embedded in v12 page.
Other user screenshot 1 and 3 both refer to BALANCED100.
"""
from pathlib import Path
import json, numpy as np, pandas as pd
from playwright.sync_api import sync_playwright

root=Path('/mnt/data'); out=root/'imba_genome_v13_work'; out.mkdir(exist_ok=True)
s=(root/'imba_genome_parameter_lab_v12_familien_transparent.html').read_text()
needle='let reserve=1000,cash=0,basis=0,paid=0,payoutN=0,stopN=0,milestone=0;'
assert s.count(needle)==1
s=s.replace(needle, 'let reserve=(window.__PORT_INIT??1000),cash=0,basis=0,paid=0,payoutN=0,stopN=0,milestone=0;')
PRESETS={
    'shot_a_balanced30_stop30':dict(pool='balanced100',rank=30,guard=30),
    'shot_b_original14_stop20':dict(pool='original462',rank=14,guard=20),
    'shot_c_balanced30_stop25':dict(pool='balanced100',rank=30,guard=25),
    'raw100_comparator_30_stop30':dict(pool='top100',rank=30,guard=30),
}
rows=[]; curves={}; validations=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-gpu','--disable-webgl'])
    page=browser.new_page(viewport={'width':1400,'height':900})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.set_content(s, wait_until='domcontentloaded',timeout=120000)
    page.wait_for_function('window.__QUANT_V12_READY===true',timeout=120000)
    for key,preset in PRESETS.items():
        for seed in ([1000,333.33,333.34] if key.startswith('shot_c_') else [1000,333.33]):
            data=page.evaluate('''({pool,rank,guard,seed}) => {
                window.__PORT_INIT=seed;state.mode=pool;state.priceMode='renko_usd';state.box='all';
                state.familyEnabled={none:true,tp:true,ttp:true,both:true};state.filtered=eligible();
                state.rank=rank;state.guard=guard;state.portKey='';state.port=null;
                return choosePortfolio().map(d=>({date:d.date,wealth:d.wealth,reserve:d.reserve,trading:d.trading,
                 active:d.live.length,selected:d.ids.filter(x=>x>=0).length,payouts:d.payoutN,stops:d.stopN}));
            }''',dict(**preset,seed=seed))
            df=pd.DataFrame(data)
            suffix=f'{int(seed)}' if seed==1000 else ('33334' if seed==333.34 else '33333')
            p=out/f'{key}_{suffix}.csv';df.to_csv(p,index=False)
            vals=df.wealth.to_numpy();rets=vals/np.r_[seed,vals[:-1]]-1
            sd=np.std(rets,ddof=1)
            stats=dict(label=key,seed=seed,**preset,end_eur=vals[-1],sharpe=np.mean(rets)/sd*np.sqrt(365) if sd>1e-12 else np.nan,
                       max_dd_pct=np.min(vals/np.maximum.accumulate(np.r_[seed,vals])[1:]-1)*100,source=p.name)
            rows.append(stats)
            curves[(key,suffix)]=df
            print(key,seed,round(vals[-1],2),round(stats['sharpe'],3),flush=True)
    assert not errors,errors
    browser.close()
stats=pd.DataFrame(rows);stats.to_csv(out/'screenshot_preset_evidence_2026.csv',index=False)
checks=[('shot_a_balanced30_stop30',7242.873165803406),('shot_b_original14_stop20',9367.04925136168),('shot_c_balanced30_stop25',10654.97)]
for n,target in checks:
    actual=stats[(stats.label==n)&(stats.seed==1000)].end_eur.iloc[0]
    assert abs(actual-target)<.03,(n,actual,target)
print('Screenshot results checked to EUR 0.03')
# 3 distinct families, each with 333.33 / 333.33 / 333.34 (balanced one receives 333.34)
# raw100 screenshot not supplied, so reference 30/30 explicitly marked as comparison assumption.
a=curves[('shot_b_original14_stop20','33333')]
b=curves[('raw100_comparator_30_stop30','33333')]
c=curves[('shot_c_balanced30_stop25','33334')]
base=pd.DataFrame({'date':a.date,'original462':a.wealth,'raw100':b.wealth,'balanced100':c.wealth})
base['mixed']=base[['original462','raw100','balanced100']].sum(axis=1)
base.to_csv(out/'screenshot_mix_2026_daily.csv',index=False)
v=base.mixed.to_numpy();ret=v/np.r_[1000,v[:-1]]-1
sharpe=np.mean(ret)/np.std(ret,ddof=1)*np.sqrt(365)
dd=np.min(v/np.maximum.accumulate(np.r_[1000,v])[1:]-1)*100
print('SCREENSHOT MIX',v[-1],sharpe,dd)