"""Additional genuinely backtested IMBA input series for v8.

Adds candle-based IMBA and dynamic percentage Renko, plus USD Renko .6-1.0.
Uses 5m confirmed OHLC for candle bounds and 5m close for Renko.
Orders/TP/TTP use v7 next-5min observation, 0.06% per order side.
Research-only; not verified TradingView parity.
"""
from pathlib import Path
import zipfile,time,numpy as np,pandas as pd,sys
from numba import njit
P=Path('/mnt/data'); W=P/'imba_genome_v8';W.mkdir(exist_ok=True)
t0=time.time()
z=zipfile.ZipFile(P/'IMBA_1x_band_weather_and_walkforward_2025_2026 (1).zip')
s=z.read('source_data/sweep_micro_imba.py').decode().replace('njit(cache=True)','njit(cache=False)')
ns={'__name__':'imba_micro_v8','__file__':str(W/'sweep_micro_v8.py')}
exec(s[:s.index('def main():')],ns)
build_fixed=ns['build_renko'];signal_from_bounds=ns['signal_from_bounds']
s7=zipfile.ZipFile(P/'IMBA_Genome_Parameter_Lab_v7_full_4158.zip').read('research/build_imba_full_genome_2026_v7.py').decode()
ns2={'np':np,'njit':njit}
exec(s7[s7.index('@njit\ndef simulate_variants'):s7.index("if __name__=='__main__':")],ns2)
simulate_variants=ns2['simulate_variants']

@njit
def build_percent_renko(price,pct):
    c=[];hi=[];lo=[];idx=[]
    last=price[0];direction=0
    for i in range(1,len(price)):
        p=price[i]
        if not np.isfinite(p) or p<=0:continue
        for j in range(400):
            step=abs(last)*pct/100.
            if step<=1e-12:break
            if direction==0:
                if p>=last+step:side=1
                elif p<=last-step:side=-1
                else:break
                nxt=last+side*step
                c.append(nxt);hi.append(max(nxt,last));lo.append(min(nxt,last));idx.append(i)
                last=nxt;direction=side
            elif direction==1:
                if p>=last+step:
                    nxt=last+step
                    c.append(nxt);hi.append(nxt);lo.append(last);idx.append(i);last=nxt
                elif p<=last-2*step:
                    nxt=last-2*step
                    c.append(nxt);hi.append(last-step);lo.append(nxt);idx.append(i);last=nxt;direction=-1
                else:break
            else:
                if p<=last-step:
                    nxt=last-step
                    c.append(nxt);hi.append(last);lo.append(nxt);idx.append(i);last=nxt
                elif p>=last+2*step:
                    nxt=last+2*step
                    c.append(nxt);hi.append(nxt);lo.append(last+step);idx.append(i);last=nxt;direction=1
                else:break
    return np.array(c),np.array(hi),np.array(lo),np.array(idx)

def main():
    print('READ PRICE',flush=True)
    # 5-minute UTC data, no intrabar execution assumptions introduced.
    a=pd.read_csv(P/'SOLUSDT (4).csv.gz',compression='gzip',sep='|',header=None,usecols=[0,2,3,4],names=['ts','hi','lo','px'],dtype={'ts':'int64','hi':'float64','lo':'float64','px':'float64'})
    a['dt']=pd.to_datetime(a.ts,unit='s',utc=True)
    a=a[(a.dt>='2021-12-01')&(a.dt<'2026-11-05')]
    bars=a.set_index('dt').resample('5min',closed='right',label='right').agg({'px':'last','hi':'max','lo':'min'}).dropna();del a
    price=bars.px.to_numpy(dtype=np.float64);bh=bars.hi.to_numpy(dtype=np.float64);bl=bars.lo.to_numpy(dtype=np.float64)
    dt=bars.index; del bars
    st=int(np.searchsorted(dt.values,np.datetime64('2026-01-01')))
    en=int(np.searchsorted(dt.values,np.datetime64('2026-11-05')))
    st25=int(np.searchsorted(dt.values,np.datetime64('2025-01-01')))
    day=np.asarray((dt.values.astype('datetime64[D]')-np.datetime64('2026-01-01')).astype('timedelta64[D]').astype('int64'),dtype=np.int64)
    day25=np.asarray((dt.values.astype('datetime64[D]')-np.datetime64('2025-01-01')).astype('timedelta64[D]').astype('int64'),dtype=np.int64)
    print('BARS',len(price),'2025-start',st25,'2026-start',st,'2026-end',en,flush=True)
    # compile JIT before full sweep
    build_fixed(np.array([10.,10.2,10.1,9.5,10.]),.1)
    build_percent_renko(np.array([10.,10.2,10.1,9.5,10.]),.1)
    simulate_variants(np.array([10.,11.,12.]),np.zeros(3,np.int8),np.zeros(3,np.int64),0,3,1)
    signal_from_bounds(np.array([10.,11.]),np.array([0,1]),np.array([11.,12.]),np.array([9.,10.]),3)
    sens=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,120]
    # new USD grid only .6-1.0; .1-.5 already available in the original 21-grid at increments .02
    boxes=[('candle',0.)]+[('renko_usd',float(x)) for x in np.round(np.arange(.6,1.01,.1),2)]+[('renko_pct',float(x)) for x in np.round(np.arange(.1,1.01,.1),2)]
    keys=['none','tp6','tp4','tp3','tp6_ttp_3_1p5','tp2','tp4_ttp_2_1','ttp_3_1p5','ttp_2_1']
    k=0
    meta=[];out=[];out25=[]
    for kind,b in boxes:
        if kind=='candle':
            c=price;h=bh;l=bl;ix=np.arange(len(price),dtype=np.int64)
        elif kind=='renko_usd':c,h,l,ix=build_fixed(price,b)
        else:c,h,l,ix=build_percent_renko(price,b)
        print('INPUT',kind,b,'BRICKS',len(c),'elapsed',round(time.time()-t0,1),flush=True)
        hs=pd.Series(h);ls=pd.Series(l)
        for S in sens:
            w=int(S*10)
            rh=hs.rolling(w,min_periods=w).max().to_numpy();rl=ls.rolling(w,min_periods=w).min().to_numpy()
            sig=signal_from_bounds(c,ix,rh,rl,len(price))
            e=simulate_variants(price,sig,day,st,en,308)
            e25=simulate_variants(price,sig,day25,st25,st,365)
            for v,key in enumerate(keys):
                ident=f'S{S}_{kind}_{b:.2f}_{key}' if kind!='candle' else f'S{S}_C5_{key}'
                meta.append((ident,S,b,key,kind));out.append(e[v].astype(np.float32));out25.append(e25[v].astype(np.float32));k+=1
        del c,h,l,ix,hs,ls
    df=pd.DataFrame(meta,columns=['strategy_id','sensitivity','renko_param','exit_key','price_mode'])
    e=np.stack(out);e25=np.stack(out25)
    assert len(df)==3168 and e.shape==(3168,308)
    df['return_2025_1x_pct']=(e25[:,-1]-1)*100;df['return_2026_1x_pct']=(e[:,-1]-1)*100
    df.to_csv(W/'extra_3168_annual_2025_2026.csv',index=False)
    np.savez_compressed(W/'extra_3168_mode_curves_2025_2026.npz',ids=df.strategy_id.to_numpy(),sens=df.sensitivity.to_numpy(),param=df.renko_param.to_numpy(),mode=df.price_mode.to_numpy(),exit_key=df.exit_key.to_numpy(),equity=e,equity_2025=e25)
    print('DONE',len(df),'t',round(time.time()-t0,1),'sec',flush=True)
if __name__=='__main__':main()