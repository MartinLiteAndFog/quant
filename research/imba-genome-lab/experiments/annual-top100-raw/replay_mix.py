"""Replay Quant Genome v12 daily 10x portfolio from embedded 1x factors.
Three independent 333.33 EUR sleeves, mix by sum, no inter-sleeve transfers.
This explicitly mirrors choosePortfolio() in imba_genome_app_v12.js.
"""
from __future__ import annotations
import base64,gzip,json,re,math,sys
from pathlib import Path
from functools import lru_cache
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
OUT=P/'results_optional'
SOURCE=P/'imba_genome_parameter_lab_v12_familien_transparent.html'
SRC=SOURCE.read_text(encoding='utf-8')

def embed(key):
    pos=SRC.index('const '+key+'=')+len('const '+key+'=')
    return json.JSONDecoder().raw_decode(SRC[pos:])[0]
META=embed('META')
PRESETS={'original462':embed('ORIGINAL'),'top100_2022':embed('RAW100'),'top100_balanced':embed('BAL100')}
YEARS=embed('YEAR_GZIP'); LENGTHS=embed('YEAR_LENGTHS')

@lru_cache(maxsize=2)
def load(year):
    n=int(LENGTHS[str(year)]); w=90+n
    raw=gzip.decompress(base64.b64decode(YEARS[str(year)]))
    fac=np.frombuffer(raw,dtype='<f4').reshape(len(META),w).astype(np.float64)
    prefix=np.empty((len(META),w+1),np.float64)
    prefix[:,0]=0.
    np.cumsum(np.log(np.maximum(fac,1e-15)),axis=1,out=prefix[:,1:])
    dates=pd.date_range(f'{year}-01-01',periods=n,freq='D')
    return fac,prefix,dates

def pre_scores(prefix,day,rank,ids):
    # exactly rPct(i,day-1,rank) in JS. day0 includes late-2025 history.
    p=90+day
    lo=max(0,p-rank)
    return 100*np.expm1(prefix[ids,p]-prefix[ids,lo])

def simulate(year,pool,rank=14,guard=30.,seed=333.33):
    fac,prefix,dates=load(year)
    ids0=np.asarray(PRESETS[pool],dtype=np.int32)
    reserve=float(seed);cash=0.;basis=0.;paid=0.;milestone=0
    payout_n=0;stop_n=0;h=np.zeros(6); chosen=np.full(6,-1,dtype=int);stopped=np.zeros(6,dtype=bool)
    idx=np.ones(6);hi=np.ones(6)
    result=[];events=[];selection=[]
    for d,date in enumerate(dates):
        new_month=date.day==1;new_cycle=d%14==0
        sub=0.
        if new_month:
            sub=min(reserve,max(100.,.2*reserve))
            reserve-=sub;cash+=sub;basis+=sub
        if new_cycle:
            cash+=sum(h);h.fill(0);chosen.fill(-1);stopped.fill(False);idx.fill(1);hi.fill(1)
            scores=pre_scores(prefix,d,rank,ids0)
            valid=np.flatnonzero(scores>0)
            order=np.lexsort((ids0[valid],-scores[valid]))[:6]
            s=ids0[valid[order]]
            chosen[:len(s)]=s
            for slot,bid in enumerate(s): selection.append(dict(date=date.date().isoformat(),slot=slot+1,bot_id=META[int(bid)]['id'],score_pct=float(scores[valid[order[slot]]])))
            if len(s) and cash>0:
                h[:len(s)]=cash/len(s);cash=0.
        elif sub>0:
            active=(chosen>=0)&~stopped
            if active.any():
                h[active]+=sub/active.sum();cash-=sub
        for k in range(6):
            if chosen[k]>=0 and not stopped[k] and h[k]>1e-10:
                f=max(0.,1.+10*(fac[chosen[k],90+d]-1.))
                h[k]*=f;idx[k]*=f;hi[k]=max(hi[k],idx[k])
        nav=cash+sum(h)
        profit=max(0,nav+paid-basis)
        ratio=profit/basis if basis>0 else 0
        if basis>0 and ratio>=max(1,milestone+1)-1e-10:
            take=min(nav*.5,.5*profit)
            if take>1e-10:
                c=min(cash,take);cash-=c;rest=take-c
                if rest>1e-10 and sum(h)>0:
                    scale=max(0,1-rest/sum(h));h*=scale
                reserve+=take;paid+=take;payout_n+=1;milestone+=1
                events.append(dict(date=date.date().isoformat(),event='profit_transfer',sum_eur=take))
        if guard>0:
            for k in range(6):
                if chosen[k]>=0 and not stopped[k] and (idx[k]/max(hi[k],1e-12)-1)<=-guard/100:
                    stopped[k]=True;cash+=h[k];h[k]=0;stop_n+=1
                    events.append(dict(date=date.date().isoformat(),event='stop',slot=k+1,bot=META[int(chosen[k])]['id']))
        nav=cash+sum(h)
        result.append(dict(date=date.date().isoformat(),wealth=reserve+nav,reserve=reserve,trading=nav,cash=cash,selected=int((chosen>=0).sum()),active=int(((chosen>=0)&(~stopped)&(h>1e-10)).sum()),stops=stop_n,payouts=payout_n,paid=paid,h=h.tolist(),ids=chosen.tolist(),stopped=stopped.tolist(),idx=idx.tolist(),hi=hi.tolist()))
    return pd.DataFrame(result),pd.DataFrame(events),pd.DataFrame(selection)

def sharpe(df):
    x=np.asarray(df['wealth'],float)
    # Includes day-1 return from initial capital, daily end-of-day, crypto 365 annualization.
    returns=np.diff(np.r_[df.attrs.get('start_capital',x[0]),x])/np.r_[df.attrs.get('start_capital',x[0]),x[:-1]]
    if np.std(returns,ddof=1)<1e-12:return np.nan
    return np.mean(returns)/np.std(returns,ddof=1)*np.sqrt(365)

def stats(df,init):
    v=df.wealth.to_numpy()
    return dict(end_eur=float(v[-1]),profit_eur=float(v[-1]-init),return_pct=float((v[-1]/init-1)*100),sharpe=sharpe(df),max_dd_pct=float(np.min(v/np.maximum.accumulate(np.r_[init,v])[1:]-1)*100))

if __name__=='__main__':
    combos=[]
    for pool in PRESETS:
        for rank in [7,10,14,30,90]:
            for guard in [0,20,30,40,50,60,80]:
                df,ev,sel=simulate(2026,pool,rank,guard,333.33)
                df.attrs['start_capital']=333.33
                s=stats(df,333.33)
                combos.append(dict(pool=pool,rank=rank,stop_pct=guard,**s))
    params=pd.DataFrame(combos)
    params.to_csv(OUT/'all_2026_settings_scans.csv',index=False)
    print('BEST per pool ex post (diagnostic, NOT validation):')
    for pool in PRESETS:
        t=params[params.pool==pool].sort_values('end_eur',ascending=False).head(8)
        print(pool,'\n',t.to_string(index=False))
    # Evidence based shortlist: common ranking windows 7/10/14, guard in 20/30/40
    print('STANDARD rank 14 stop30:')
    print(params[(params['rank']==14)&(params['stop_pct']==30)].to_string(index=False))