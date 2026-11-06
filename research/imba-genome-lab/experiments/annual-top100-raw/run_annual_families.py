"""Causal annual Top100 Raw family walkforward on verified v12 2022-26 daily factor dataset.
Same original Top100 generation as year 2022: 462 fixed-USD Renko x 9 exit genes.
Portfolio matches v12 choosePortfolio daily ledger, 10x daily return, monthly reserve transfer,
14-day portfolio rotation, positive-only momentum, individually stopped bots, milestone payouts.
No lookahead. Research-only; no intraday liquidation or exchange funding.
"""
from pathlib import Path
import numpy as np, pandas as pd, sys, math, itertools
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_mix import load, META, PRESETS, pre_scores
BASE=Path(__file__).resolve().parent;BASE.mkdir(parents=True,exist_ok=True)
IDS=np.array([i for i,m in enumerate(META) if m['priceMode']=='renko_usd' and m['b']<=.500000001],dtype=np.int32)
ORIG462=np.asarray(PRESETS['original462'],dtype=np.int32)
assert len(IDS)==4158 and len(ORIG462)==462

def family_of(year,ret=None):
    if ret is None:
        fac,pref,dates=load(year)
        # annual pure 1x compounded return
        ret=100*(np.prod(fac[IDS,90:],axis=1)-1)
    ranked=IDS[np.argsort(-ret,kind='stable')]
    return ranked[:100]
COHORT={y:family_of(y) for y in range(2022,2026)}
assert set(COHORT[2022])==set(PRESETS['top100_2022']),'2022 parity with archived raw100 failed'

cohort_rows=[];shares=[]
for y,arr in COHORT.items():
    fac,pref,dates=load(y)
    rr=100*(np.prod(fac[IDS,90:],axis=1)-1.)
    order=IDS[np.argsort(-rr,kind='stable')]
    lookup=dict(zip(IDS,rr))
    for rank,i in enumerate(arr,1):
        m=META[int(i)];cohort_rows.append(dict(training_year=y,rank=rank,bot_index=int(i),strategy_id=m['id'],price_mode=m['priceMode'],sensitivity=m['s'],renko_usd=m['b'],exit_key=m['exitKey'],exit_family=m['family'],return_1x_train_pct=lookup[i]))
    cnt=pd.Series([META[i]['family'] for i in arr]).value_counts()
    shares.append({'training_year':y,'profitable_of_4158':int(np.sum(rr>0)),'best_2022_ref_overlap':int(len(set(arr)&set(COHORT[2022]))),'train_top100_min_pct':float(min(lookup[i] for i in arr)),'train_top100_max_pct':float(max(lookup[i] for i in arr)),**{k:int(cnt.get(k,0)) for k in ['none','tp','ttp','both']}})
pd.DataFrame(cohort_rows).to_csv(BASE/'annual_top100_raw_families_2022_2025.csv',index=False)
pd.DataFrame(shares).to_csv(BASE/'annual_top100_genes_summary.csv',index=False)


def group_for(year,method):
    if method=='carry_2022':return COHORT[2022]
    if method=='original462':return ORIG462
    if method=='annual_rotate':return COHORT[year-1]
    if method=='accumulate_past':return np.array(list(dict.fromkeys(int(i) for y in range(2022,year) for i in COHORT[y])),dtype=np.int32)
    if method.startswith('cohort_'):
        cohort=int(method.split('_')[1]);return COHORT[cohort]
    raise ValueError(method)


def simulate(year_pool,rank=14,guard=30.,initial=1000.,correlation=None,fallback=True,record_daily=True):
    """year_pool: iterable (calendar year, candidate indexes), ordered sequentially.

    Handles one continuous treasury across years, with per-year new cycle reset.
    """
    reserve=float(initial);cash=basis=paid=0.
    payoutN=stopN=milestone=0
    h=np.zeros(6,float); chosen=np.full(6,-1,dtype=int);stopped=np.zeros(6,bool);idx=np.ones(6);hi=np.ones(6)
    rows=[];sels=[];events=[]
    for year,allow in year_pool:
        f,prefix,dates=load(year)
        allow=np.asarray(allow,dtype=np.int32)
        assert len(allow)>0
        for d,date in enumerate(dates):
            today=date.strftime('%Y-%m-%d')
            sub=0.
            if date.day==1:
                sub=min(reserve,max(100.,0.2*reserve));reserve-=sub;cash+=sub;basis+=sub
            if d%14==0:
                cash+=float(h.sum());h.fill(0);chosen.fill(-1);stopped.fill(False);idx.fill(1);hi.fill(1)
                scores=pre_scores(prefix,d,rank,allow)
                valid=np.flatnonzero(scores>0.)
                order=allow[valid[np.lexsort((allow[valid],-scores[valid]))]]
                selected=[]
                fallback_used=0
                if correlation is None:
                    selected=list(map(int,order[:6]))
                else:
                    # <= max pairwise positive correlation over past 28 completed 1x days.
                    end=90+d;beg=end-28
                    r=f[order,beg:end]-1.
                    c=r-r.mean(axis=1,keepdims=True)
                    norms=np.linalg.norm(c,axis=1)
                    normalized=np.divide(c,norms[:,None],out=np.zeros_like(c),where=(norms[:,None]>1e-10))
                    ix=[]
                    for j,bot in enumerate(order):
                        if len(selected)>=6:break
                        if norms[j]<1e-10:continue
                        if all(float(np.dot(normalized[j],normalized[z]))<=correlation+1e-12 for z in ix):
                            selected.append(int(bot));ix.append(j)
                    if fallback and len(selected)<6:
                        for bot in order:
                            if len(selected)>=6:break
                            if int(bot) not in selected:
                                selected.append(int(bot));fallback_used+=1
                chosen[:len(selected)]=selected
                if selected and cash>0:
                    h[:len(selected)]=cash/len(selected);cash=0.
                for k,bot in enumerate(selected):
                    sels.append(dict(year=year,date=today,slot=k+1,bot_id=META[bot]['id'],bot_index=bot,training_year=('previous' if isinstance(year_pool,str) else year-1),past_rank_pct=float(pre_scores(prefix,d,rank,np.array([bot]))[0]),corr_fallback=fallback_used))
            elif sub>0:
                active=(chosen>=0)&~stopped
                if active.any():h[active]+=sub/int(active.sum());cash-=sub
            for k in range(6):
                if chosen[k]>=0 and not stopped[k] and h[k]>1e-10:
                    mult=max(0.,1.+10*(f[int(chosen[k]),90+d]-1.))
                    h[k]*=mult;idx[k]*=mult;hi[k]=max(hi[k],idx[k])
            nav=cash+float(h.sum())
            profit=max(0.,nav+paid-basis)
            ratio=profit/basis if basis>0 else 0.
            if basis>0 and ratio>=max(1,milestone+1)-1e-10:
                take=min(nav*.5,.5*profit)
                if take>1e-10:
                    used=min(cash,take);cash-=used;rest=take-used
                    if rest>1e-10 and h.sum()>0:h*=max(0.,1.-rest/h.sum())
                    reserve+=take;paid+=take;payoutN+=1;milestone+=1
                    events.append(dict(year=year,date=today,event='profit_transfer',amount_eur=take))
            if guard>0:
                for k in range(6):
                    if chosen[k]>=0 and not stopped[k] and (idx[k]/max(hi[k],1e-12)-1.)<=-guard/100.:
                        stopped[k]=True;cash+=h[k];h[k]=0.;stopN+=1
                        events.append(dict(year=year,date=today,event='bot_stop',bot_id=META[int(chosen[k])]['id'],slot=k+1))
            nav=cash+float(h.sum())
            rows.append(dict(year=year,date=today,wealth=reserve+nav,reserve=reserve,trading=nav,active=int(((chosen>=0)&~stopped&(h>1e-10)).sum()),selected=int((chosen>=0).sum()),payouts=payoutN,stops=stopN,paid=paid,allocated=basis))
    return pd.DataFrame(rows),pd.DataFrame(sels),pd.DataFrame(events)

def stats(df,initial=1000.):
    nav=df.wealth.to_numpy(float)
    rets=nav/np.r_[initial,nav[:-1]]-1.
    std=rets.std(ddof=1)
    sharpe=rets.mean()/std*np.sqrt(365) if std>1e-12 else np.nan
    max_dd=np.min(nav/np.maximum.accumulate(np.r_[initial,nav])[1:]-1)*100
    return dict(final_eur=float(nav[-1]),profit_eur=float(nav[-1]-initial),return_pct=float((nav[-1]/initial-1)*100),sharpe=float(sharpe),max_dd_pct=float(max_dd),trading_days=int((df.active>0).sum()),n_payouts=int(df.payouts.iloc[-1]),n_stops=int(df.stops.iloc[-1]))

if __name__=='__main__':
    summary=[];lines=[];yearly=[];selected=[]
    # individual independent cohort performance by actual future date
    for year in [2023,2024,2025,2026]:
        methods=['original462','carry_2022','annual_rotate','accumulate_past']
        for training in range(2023,year):methods.append('cohort_'+str(training))
        for method in methods:
            pop=group_for(year,method)
            for guard,rank in [(30,14),(30,30)]:
                r,s,e=simulate([(year,pop)],rank=rank,guard=guard)
                st=stats(r)
                summary.append(dict(year=year,method=method,pool_size=len(pop),score_days=rank,stop_pct=guard,**st))
                if rank==14:
                    lines.extend([dict(method=method,rank=rank,year=year,date=row.date,wealth=row.wealth) for row in r.itertuples(index=False)])
                    if method in ('annual_rotate','carry_2022','accumulate_past'):
                        selected.extend([dict(method=method,**row) for row in s.to_dict('records')])

    # Chained 2023->2026: one 1000 euro starting depot, family policy changes each year
    for method in ['carry_2022','annual_rotate','accumulate_past','original462']:
        for guard,rank in [(30,14),(30,30)]:
            yp=[(year,group_for(year,method)) for year in range(2023,2027)]
            r,s,e=simulate(yp,rank=rank,guard=guard)
            for y,g in r.groupby('year'):
                yearly.append(dict(year=int(y),method=method,score_days=rank,stop_pct=guard,year_end_eur=float(g.wealth.iloc[-1]),n_available=len(group_for(int(y),method))))
            if rank==14:r.assign(method=method,score_days=rank,stop_pct=guard).to_csv(BASE/f'chained_{method}_2023_2026_daily.csv',index=False)

    pd.DataFrame(summary).to_csv(BASE/'independent_yearly_cohort_comparisons.csv',index=False)
    pd.DataFrame(lines).to_csv(BASE/'individual_equity_curves_2023_2026.csv',index=False)
    pd.DataFrame(yearly).to_csv(BASE/'chained_2023_2026_comparison.csv',index=False)
    pd.DataFrame(selected).to_csv(BASE/'annual_rotation_top6_selections.csv',index=False)
    print('FAMILY_GENE_BREAKDOWN')
    print(pd.DataFrame(shares).to_string(index=False))
    print('\nINDEPENDENT YEARLY rank14 stop30:')
    print(pd.DataFrame(summary).query('score_days==14').pivot(index='year',columns='method',values='final_eur').round(2).to_string())
    print('\nINDEPENDENT YEARLY rank30 stop30:')
    print(pd.DataFrame(summary).query('score_days==30').pivot(index='year',columns='method',values='final_eur').round(2).to_string())
    print('\nCHAINED 14:')
    print(pd.DataFrame(yearly).query('score_days==14').pivot(index='year',columns='method',values='year_end_eur').round(2).to_string())