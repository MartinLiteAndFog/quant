"""Browser-faithful annual cohort rotation on Quant v12 embedded year factors.
Reuses JS choosePortfolio and selection, prevents floating tie drift of NumPy.
"""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_annual_families import COHORT, group_for, ORIG462

out=Path(__file__).resolve().parent
html=(out/'imba_genome_parameter_lab_v12_familien_transparent.html').read_text(encoding='utf8')
methods=['original462','carry_2022','annual_rotate','accumulate_past']
results=[];daily=[];top6=[]
with sync_playwright() as pw:
    b=pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-gpu','--disable-webgl'])
    page=b.new_page(viewport={'width':1280,'height':820})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.set_content(html,wait_until='domcontentloaded',timeout=120000)
    page.wait_for_function('window.__QUANT_V12_READY===true',timeout=120000)
    for year in [2023,2024,2025,2026]:
        if year !=2026:page.evaluate('async y => await loadYear(String(y))',year)
        else:page.evaluate('async y => await loadYear(String(y))',year)
        these=methods+['cohort_'+str(t) for t in range(2023,year)]
        for method in these:
            eligible=list(map(int,group_for(year,method)))
            for rank in [14,30]:
                x=page.evaluate('''({pool,rank})=>{
                  state.mode='manual';state.manual=new Set(pool);
                  state.priceMode='renko_usd';state.box='all';state.familyEnabled={none:true,tp:true,ttp:true,both:true};
                  state.guard=30;state.rank=rank;state.filtered=eligible();state.portKey='';state.port=null;
                  const res=choosePortfolio();
                  return res.map((d,j)=>({date:d.date,wealth:d.wealth,reserve:d.reserve,trading:d.trading,
                      active:d.live.length,ids:(j%14===0)?d.ids.filter(q=>q>=0):null,
                      payoutN:d.payoutN,stopN:d.stopN}));
                }''',{'pool':eligible,'rank':rank})
                wealth=np.array([t['wealth'] for t in x],float)
                initial=1000.0;returns=wealth/np.r_[initial,wealth[:-1]]-1.0
                std=np.std(returns,ddof=1)
                results.append(dict(year=year,method=method,score_days=rank,stop_pct=30,pool_size=len(eligible),final_eur=wealth[-1],profit_pct=(wealth[-1]/initial-1)*100,sharpe=returns.mean()/std*np.sqrt(365) if std>1e-12 else np.nan,drawdown_pct=(wealth/np.maximum.accumulate(np.r_[initial,wealth])[1:]-1).min()*100,trade_days=sum(q['active']>0 for q in x),payouts=x[-1]['payoutN'],stops=x[-1]['stopN']))
                for item in x:
                    if rank==14:daily.append(dict(year=year,method=method,**{k:v for k,v in item.items() if k!='ids'}))
                    if item['ids'] is not None:top6.extend([dict(year=year,method=method,rank_days=rank,date=item['date'],slot=j+1,bot_index=i) for j,i in enumerate(item['ids'])])
                print('BROWSER',year,method,'rank',rank,'end',round(wealth[-1],3),flush=True)
    print('JAVASCRIPT_ERRORS',errors,flush=True)
    assert not errors, errors
    b.close()
res=pd.DataFrame(results);res.to_csv(out/'browser_verified_independent_years.csv',index=False)
pd.DataFrame(daily).to_csv(out/'browser_verified_daily_equity.csv',index=False)
pd.DataFrame(top6).to_csv(out/'browser_verified_top6_selection.csv',index=False)
print('14days\n',res.query('score_days==14').pivot(index='year',columns='method',values='final_eur').round(2).to_string())
print('30days\n',res.query('score_days==30').pivot(index='year',columns='method',values='final_eur').round(2).to_string())