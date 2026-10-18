"""Independent traditional Renko and HTF expressions from minute market OHLC.

Reference CSV is consumed ONLY by the evaluator in __main__, not construction.
Regular interval closes become available at interval end; synthetic chart times
remain separate and must not be mistaken for executable order times.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ew_replay import atr_series, pivot_series


def aggregate_minutes(minutes, seconds, input_seconds=60):
    groups=minutes.groupby(minutes.time//seconds,sort=True)
    bars=groups.agg(time=('time','first'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),samples=('time','size'))
    bars['time']=bars.index.to_numpy()*seconds
    return bars.loc[bars.samples==seconds//input_seconds].reset_index(drop=True)


def build_renko(bars, seconds, box=.1, wick_mode='close'):
    if box<=0 or wick_mode not in ('close','ohlc','none'):
        raise ValueError('Invalid Renko specification')
    last=int(np.floor(float(bars.close.iloc[0])/box+1e-8)); direction=0
    pending_high=last*box; pending_low=last*box; records=[]
    for row in bars.itertuples(index=False):
        price=float(row.close)
        pending_high=max(pending_high,float(row.high) if wick_mode=='ohlc' else price)
        pending_low=min(pending_low,float(row.low) if wick_mode=='ohlc' else price)
        up=int(np.floor(price/box+1e-8))-last
        down=last-int(np.ceil(price/box-1e-8))
        newdir=1 if up >= (2 if direction==-1 else 1) else -1 if down >= (2 if direction==1 else 1) else 0
        if not newdir: continue
        reverse=direction!=0 and direction!=newdir
        if reverse: last+=newdir
        count=(up if newdir==1 else down)-int(reverse)
        for seq in range(count):
            op=last*box; last+=newdir; cl=last*box
            hi=max(op,cl); lo=min(op,cl)
            if seq==0 and wick_mode!='none':
                if newdir==1 and pending_low<op-box-1e-8: lo=min(lo,pending_low)
                elif newdir==-1 and pending_high>op+box+1e-8: hi=max(hi,pending_high)
            available=int(getattr(row,'available_ms',(int(row.time)+seconds)*1000))
            records.append((int(row.time)*1000+seq,available,op,hi,lo,cl))
        direction=newdir; pending_high=last*box; pending_low=last*box
    return pd.DataFrame(records,columns=['time_ms','available_ms','open','high','low','close'])


def htf_expression(renko, length, atr_length=14):
    n=len(renko)
    out=pd.DataFrame({'time_ms':renko.time_ms,'available_ms':renko.available_ms.shift(1)})
    for col,hi in [('high',True),('low',False)]:
        raw=pivot_series(renko[col].to_numpy(),length,hi)
        out['pivot '+col+' raw']=pd.Series(raw).shift(1)
    out['ATR']=pd.Series(atr_series(renko,atr_length)).shift(1)
    out['pivot time']=renko.time_ms.shift(length+1)
    out['pivot idx']=pd.Series(np.arange(n)).shift(length+1)
    return out


def map_expression(expression, chart_times):
    """Historical lookahead-on mapping candidate, separately audited for causality."""
    return pd.merge_asof(pd.DataFrame({'time_ms':np.asarray(chart_times,dtype=np.int64)}),
                         expression.sort_values('time_ms'),on='time_ms',direction='backward')


def confirmed_htf_state(renko, length, chart_available_ms, atr_length=14):
    """Last completed HTF state; causal alternative, not historical lookahead.

    Build directly at each completed brick, including the last brick of the data
    prefix. Do not wait for a future brick to materialize a shifted expression.
    """
    state=pd.DataFrame({'available_ms':renko.available_ms})
    for col,hi in [('high',True),('low',False)]:
        state['pivot '+col+' raw']=pivot_series(renko[col].to_numpy(),length,hi)
    state['ATR']=atr_series(renko,atr_length)
    state['pivot time']=renko.time_ms.shift(length)
    state['pivot idx']=pd.Series(np.arange(len(renko))).shift(length)
    state=state.drop_duplicates('available_ms',keep='last')
    state['source_available_ms']=state.available_ms
    mapped=pd.merge_asof(pd.DataFrame({'available_ms':np.asarray(chart_available_ms,dtype=np.int64)}),state,on='available_ms',direction='backward')
    mapped['available_ms']=mapped.pop('source_available_ms')
    return mapped


def evaluate_raw(reference, predicted, k, mask):
    results={}
    for col in ['pivot high raw','pivot low raw','ATR','pivot time']:
        actual=reference[f'Audit htf{k} '+col].to_numpy(float)
        pred=predicted[col].to_numpy(float)
        ok=np.isclose(actual,pred,rtol=0,atol=1e-8,equal_nan=True)
        nonempty=mask & np.isfinite(actual)
        results[col]={'rows':int(mask.sum()),'match':int((ok&mask).sum()),
                      'reference_nonempty':int(nonempty.sum()),'nonempty_match':int((ok&nonempty).sum())}
    return results


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    minutes=pd.read_csv(root/'inputs/market_minutes.csv.gz')
    ref=pd.read_csv(root/'inputs/BINANCE_SOLUSDT, 5 (5).csv',float_precision='round_trip')
    times=np.rint(ref.time.to_numpy()*1000).astype(np.int64)
    reports=[]
    for k,hours,length in [(1,1,8),(2,4,7)]:
        bars=aggregate_minutes(minutes,hours*3600)
        for wick in ['none','close','ohlc']:
            renko=build_renko(bars,hours*3600,wick_mode=wick)
            expr=htf_expression(renko,length)
            mapped=map_expression(expr,times)
            mask=(times>=pd.Timestamp('2026-09-07T00:00:00Z').value//1000000)&(times<(int(bars.time.max())+hours*3600)*1000)
            report={'htf':hours,'wick':wick,'bricks':len(renko),'scores':evaluate_raw(ref,mapped,k,mask)}
            reports.append(report)
            renko.to_csv(root/f'results/independent_{hours}h_{wick}_renko.csv',index=False)
            mapped.to_csv(root/f'results/independent_{hours}h_{wick}_raw.csv',index=False)
    (root/'results/independent_htf_report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))
