"""Research-only port of Pine v3.2 primary/secondary ZigZag and raw W3 setup.

Deliberately does not implement full pattern detection, HTF request.security, WC, or orders.
Uses ORIGINAL TradingView exported Renko bars to isolate Pine logic from Renko builder.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def pine_pivots(high: np.ndarray, low: np.ndarray, length: int):
    n = len(high)
    h = np.zeros(n, dtype=bool)
    l = np.zeros(n, dtype=bool)
    for idx in range(length*2,n):
        k=idx-length
        # Pine prefers latest pivot on equality: older equal OK, newer equal invalid.
        wh=high[k-length:idx+1]
        wl=low[k-length:idx+1]
        h[idx]=np.all(wh[:length] <= high[k]) and np.all(wh[length+1:] < high[k])
        l[idx]=np.all(wl[:length] >= low[k]) and np.all(wl[length+1:] > low[k])
    return h,l


def atr_wilder(high, low, close, length):
    n=len(close)
    prev=np.r_[close[0],close[:-1]]
    tr=np.maximum.reduce([high-low,np.abs(high-prev),np.abs(low-prev)])
    atr=np.full(n,np.nan)
    if n >= length:
        atr[length-1]=np.mean(tr[:length])
        for k in range(length,n):
            atr[k]=(atr[k-1]*(length-1)+tr[k])/length
    return atr


def port_wave3(df: pd.DataFrame,pri_len: int=13,sec_len:int=5,atr_len:int=14,pri_mult:float=3,sec_mult:float=1.5):
    high=df.high.to_numpy(dtype=float);low=df.low.to_numpy(dtype=float);close=df.close.to_numpy(dtype=float)
    atr=atr_wilder(high,low,close,atr_len)
    ph,pl=pine_pivots(high,low,pri_len)
    sh,sl=pine_pivots(high,low,sec_len)
    primary=[];secondary=[]
    pushed=np.zeros(len(df),dtype=bool)
    w3raw=np.zeros(len(df),dtype='int8')
    w3details=[]
    def add_pivot(arr,price,is_high,idx,current_atr,mult):
        if not arr:
            arr.append([idx,price,is_high]);return True
        last=arr[-1]
        if last[2]!=is_high:
            if np.isfinite(current_atr) and abs(price-last[1]) >= current_atr*mult - 1e-10:
                arr.append([idx,price,is_high]);return True
        elif (price>last[1] if is_high else price<last[1]):
            arr[-1]=[idx,price,is_high]
        return False
    for i in range(len(df)):
        if np.isnan(atr[i]):continue
        pushed_now=False
        for is_high,yes in ((True,ph[i]),(False,pl[i])):
            if yes:
                price=high[i-pri_len] if is_high else low[i-pri_len]
                pushed_now |= add_pivot(primary,price,is_high,i-pri_len,atr[i],pri_mult)
        for is_high,yes in ((True,sh[i]),(False,sl[i])):
            if yes:
                price=high[i-sec_len] if is_high else low[i-sec_len]
                add_pivot(secondary,price,is_high,i-sec_len,atr[i],sec_mult)
        if len(primary)>100:primary.pop(0)
        if len(secondary)>300:secondary.pop(0)
        pushed[i]=pushed_now
        if pushed_now and len(primary)>=3:
            a,b,c=primary[-3:];bull = a[1] < b[1]
            ratio=abs(c[1]-b[1])/abs(b[1]-a[1]) if b[1]!=a[1] else 0
            holds=(c[1]>a[1]) if bull else (c[1]<a[1])
            subcount=sum(a[0]<p[0]<b[0] for p in secondary)
            if .382 <= ratio <= .786 and holds and subcount>=4:
                w3raw[i]=1 if bull else -1
                w3details.append({'index':i,'side':'long' if bull else 'short','pivot_a':a[0],'pivot_b':b[0], 'pivot_c':c[0], 'retracement':ratio,'secondary_internal_pivots':subcount})
    return pushed,w3raw,w3details


def score(expected:np.ndarray, actual:np.ndarray, start:int):
    e=set(np.flatnonzero(expected[start:])+start);a=set(np.flatnonzero(actual[start:])+start)
    return {'expected':len(e),'predicted':len(a),'exact_matches':len(e&a),'false_positives':len(a-e),'misses':len(e-a),'precision':round(len(a&e)/len(a),4) if a else None,'recall':round(len(a&e)/len(e),4) if e else None}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--csv',default='/mnt/data/BINANCE_SOLUSDT, 5 (5)(1).csv');args=ap.parse_args()
    df=pd.read_csv(args.csv)
    print('total',len(df))
    for opts in ((13,5,14),(11,5,13)):
        push,w3,details=port_wave3(df,*opts)
        print('settings primary, secondary, atr:',opts)
        if 'Audit primary pushed' in df:
            print('PRIMARY PUSH:',score(df['Audit primary pushed'].to_numpy()!=0,push,100))
        if 'Long Entry' in df:
            target=(df['Long Entry'].to_numpy()!=0).astype(int)-(df['Short Entry'].to_numpy()!=0).astype(int)
            print('W3 LONG',score(target==1,w3==1,100),'W3 SHORT',score(target==-1,w3==-1,100)); print('raw w3',len(details),'samples',details[:5])