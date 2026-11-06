"""Continuous 2023-2026 wealth across annual strategy-family refreshes in JS.
Exact daily logic of v12 choosePortfolio, with carried treasury state across years.
No reset of wealth; Jan-1 resets only selected strategies/cycle and lookback model.
"""
from pathlib import Path
import sys, numpy as np, pandas as pd
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_annual_families import group_for

out=Path(__file__).resolve().parent
html=(out/'imba_genome_parameter_lab_v12_familien_transparent.html').read_text(encoding='utf8')
js=r'''({pool,rank,guard,initial})=>{
 const sum=a=>a.reduce((x,y)=>x+y,0);
 if(!window.__CHAIN || initial){
   window.__CHAIN={reserve:1000,cash:0,basis:0,paid:0,payoutN:0,stopN:0,milestone:0,
   h:[0,0,0,0,0,0],ids:[-1,-1,-1,-1,-1,-1],stopped:[false,false,false,false,false,false],
   idx:[1,1,1,1,1,1],hi:[1,1,1,1,1,1]};
 }
 state.rank=rank;
 const c=window.__CHAIN,results=[];
 for(let d=0;d<DAYS;d++){
  const date=DATES[d],newMonth=date.endsWith('-01'),newCycle=d%14===0;
  let sub=0;
  if(newMonth){sub=Math.min(c.reserve,Math.max(100,.2*c.reserve));c.reserve-=sub;c.cash+=sub;c.basis+=sub;}
  if(newCycle){c.cash+=sum(c.h);c.h.fill(0);c.ids.fill(-1);c.stopped.fill(false);c.idx.fill(1);c.hi.fill(1);
   const scored=pool.map(i=>[i,score(i,d)]).filter(v=>v[1]>0).sort((a,b)=>b[1]-a[1]||a[0]-b[0]).slice(0,6);
   scored.forEach((v,k)=>c.ids[k]=v[0]);const n=scored.length;
   if(n&&c.cash>0){for(let k=0;k<n;k++)c.h[k]=c.cash/n;c.cash=0;}
  }else if(sub>0){const active=c.ids.filter((i,k)=>i>=0&&!c.stopped[k]).length;
   if(active){for(let k=0;k<6;k++)if(c.ids[k]>=0&&!c.stopped[k])c.h[k]+=sub/active;c.cash-=sub;}
  }
  for(let k=0;k<6;k++)if(c.ids[k]>=0&&!c.stopped[k]&&c.h[k]>1e-10){
   const factor=Math.max(0,1+10*(dayFactor(c.ids[k],d)-1));c.h[k]*=factor;c.idx[k]*=factor;c.hi[k]=Math.max(c.hi[k],c.idx[k]);
  }
  let nav=c.cash+sum(c.h),profit=Math.max(0,nav+c.paid-c.basis),ratio=c.basis>0?profit/c.basis:0;
  if(c.basis>0&&ratio>=Math.max(1,c.milestone+1)-1e-10){const take=Math.min(nav*.5,.5*profit);
   if(take>1e-10){const x=Math.min(c.cash,take);c.cash-=x;const rest=take-x;
    if(rest>1e-10&&sum(c.h)>0){const scale=Math.max(0,1-rest/sum(c.h));for(let k=0;k<6;k++)c.h[k]*=scale;}
    c.reserve+=take;c.paid+=take;c.payoutN++;c.milestone++;}
  }
  if(guard>0){for(let k=0;k<6;k++)if(c.ids[k]>=0&&!c.stopped[k]&&c.idx[k]/Math.max(c.hi[k],1e-12)-1<=-guard/100){
    c.stopped[k]=true;c.cash+=c.h[k];c.h[k]=0;c.stopN++;
  }}
  const trading=c.cash+sum(c.h),live=c.ids.filter((i,k)=>i>=0&&!c.stopped[k]&&c.h[k]>1e-10);
  results.push({date,wealth:c.reserve+trading,reserve:c.reserve,trading,active:live.length,payoutN:c.payoutN,stopN:c.stopN});
 }
 return results;
}'''

rows=[];checks=[]
with sync_playwright() as pw:
  b=pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-gpu','--disable-webgl'])
  page=b.new_page();page.set_content(html,wait_until='domcontentloaded',timeout=120000)
  page.wait_for_function('window.__QUANT_V12_READY===true',timeout=120000)
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  for rank in [14,30]:
    for method in ['carry_2022','annual_rotate','accumulate_past','original462']:
      for year in [2023,2024,2025,2026]:
        page.evaluate('async y => await loadYear(String(y))',year)
        pool=[int(i) for i in group_for(year,method)]
        result=page.evaluate(js,{'pool':pool,'rank':rank,'guard':30,'initial':year==2023})
        for a in result:
          rows.append(dict(year=year,method=method,score_days=rank,stop_pct=30,**a))
        checks.append(dict(year=year,method=method,score_days=rank,yearend_eur=float(result[-1]['wealth']),reserves=float(result[-1]['reserve']),payouts=int(result[-1]['payoutN']),stops=int(result[-1]['stopN'])))
        print('CHAIN',method,rank,year,'end',round(result[-1]['wealth'],3),flush=True)
  print('JS ERRORS',errors)
  assert not errors
  b.close()
pd.DataFrame(rows).to_csv(out/'browser_verified_chained_daily.csv',index=False)
pd.DataFrame(checks).to_csv(out/'browser_verified_chained_yearends.csv',index=False)