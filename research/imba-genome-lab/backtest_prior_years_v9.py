"""Backtest 2021-Dec/2022/2023/2024 for all 7326 Genome Lab v8 variants.
Uses identical source code for 5m Renko/Candle signals and close-based TP/TTP as v7/v8,
with 0.06% fee/side. Independent flat reset on each calendar period.
Research model; NOT TradingView or archived-462 signal parity.
"""
from pathlib import Path
import zipfile,time,numpy as np,pandas as pd,sys
from numba import njit
P=Path('/mnt/data'); W=P/'imba_genome_v9'; W.mkdir(exist_ok=True)
t0=time.time()
z=zipfile.ZipFile(P/'IMBA_1x_band_weather_and_walkforward_2025_2026 (1).zip')
s=z.read('source_data/sweep_micro_imba.py').decode().replace('njit(cache=True)','njit(cache=False)')
ns={'__name__':'imba_micro_v9','__file__':str(W/'sweep_micro_v9.py')}
exec(s[:s.index('def main():')],ns)
build_fixed=ns['build_renko'];signal_from_bounds=ns['signal_from_bounds']
s7=zipfile.ZipFile(P/'IMBA_Genome_Parameter_Lab_v7_full_4158.zip').read('research/build_imba_full_genome_2026_v7.py').decode()
ns2={'np':np,'njit':njit}
exec(s7[s7.index('@njit\ndef simulate_variants'):s7.index("if __name__=='__main__':")],ns2)
simulate_variants=ns2['simulate_variants']
s8=(P/'imba_genome_v8/build_new_modes.py').read_text()
ns3={'np':np,'njit':njit}
exec(s8[s8.index('@njit\ndef build_percent_renko'):s8.index('def main():', s8.index('@njit\ndef build_percent_renko'))],ns3)
build_percent_renko=ns3['build_percent_renko']

print('READ SOURCE',flush=True)
a=pd.read_csv(P/'SOLUSDT (4).csv.gz',compression='gzip',sep='|',header=None,usecols=[0,2,3,4],names=['ts','hi','lo','px'],dtype={'ts':'int64','hi':'float64','lo':'float64','px':'float64'})
a['dt']=pd.to_datetime(a.ts,unit='s',utc=True)
a=a[(a.dt>='2021-12-01')&(a.dt<'2025-01-01')]
bars=a.set_index('dt').resample('5min',closed='right',label='right').agg({'px':'last','hi':'max','lo':'min'}).dropna();del a
price=bars.px.to_numpy(dtype=np.float64);bh=bars.hi.to_numpy(dtype=np.float64);bl=bars.lo.to_numpy(dtype=np.float64)
dt=bars.index
print('5m bars',len(price),'range',str(dt[0]),str(dt[-1]),'load secs',round(time.time()-t0,2),flush=True)
periods=[]
for startstr,endstr in [('2021-12-01','2022-01-01'),('2022-01-01','2023-01-01'),('2023-01-01','2024-01-01'),('2024-01-01','2025-01-01')]:
    a=int(np.searchsorted(dt.values,np.datetime64(startstr)))
    b=int(np.searchsorted(dt.values,np.datetime64(endstr)))
    days=(pd.Timestamp(endstr)-pd.Timestamp(startstr)).days
    di=np.asarray((dt.values.astype('datetime64[D]')-np.datetime64(startstr)).astype('timedelta64[D]').astype('int64'),dtype=np.int64)
    print('period',startstr,a,b,days,flush=True)
    periods.append((startstr,a,b,days,di))

build_fixed(np.array([10.,10.2,10.1,9.5,10.]),.1)
build_percent_renko(np.array([10.,10.2,10.1,9.5,10.]),.1)
simulate_variants(np.array([10.,11.,12.]),np.zeros(3,np.int8),np.zeros(3,np.int64),0,3,1)

sens=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,120]
boxes=[('renko_usd',float(x)) for x in np.round(np.arange(.10,.501,.02),2)]
boxes += [('candle',0.)]+[('renko_usd',float(x)) for x in np.round(np.arange(.6,1.01,.1),2)]+[('renko_pct',float(x)) for x in np.round(np.arange(.1,1.01,.1),2)]
keys=['none','tp6','tp4','tp3','tp6_ttp_3_1p5','tp2','tp4_ttp_2_1','ttp_3_1p5','ttp_2_1']
year_arrays={k:np.empty((7326,d),dtype=np.float32) for k,_,_,d,_ in periods}
meta=[];pos=0
for kind,b in boxes:
    if kind=='candle':c=price;h=bh;l=bl;ix=np.arange(len(price),dtype=np.int64)
    elif kind=='renko_usd':c,h,l,ix=build_fixed(price,b)
    else:c,h,l,ix=build_percent_renko(price,b)
    hs=pd.Series(h);ls=pd.Series(l)
    print('input',kind,b,'bricks',len(c),'elapsed',round(time.time()-t0,1),flush=True)
    for S in sens:
        w=S*10
        rh=hs.rolling(w,min_periods=w).max().to_numpy();rl=ls.rolling(w,min_periods=w).min().to_numpy()
        sig=signal_from_bounds(c,ix,rh,rl,len(price))
        rows={key:simulate_variants(price,sig,di,a,z,days) for key,a,z,days,di in periods}
        for v,key in enumerate(keys):
            ident=f'S{S}_R{b:.2f}_{key}' if kind=='renko_usd' and b<=.5 else f'S{S}_{kind}_{b:.2f}_{key}' if kind!='candle' else f'S{S}_C5_{key}'
            meta.append((ident,S,b,key,kind))
            for periodname,_,_,_,_ in periods: year_arrays[periodname][pos]=rows[periodname][v].astype(np.float32)
            pos+=1
    if kind!='candle':del c,h,l,ix
    del hs,ls
assert pos==7326
meta=pd.DataFrame(meta,columns=['id','s','b','exitKey','priceMode'])
z7=np.load(P/'imba_genome_v7/full_4158_genomes_2026_daily.npz',allow_pickle=True)
z8=np.load(P/'imba_genome_v8/extra_3168_mode_curves_2025_2026.npz',allow_pickle=True)
ref_ids=np.concatenate([z7['ids'].astype(str),z8['ids'].astype(str)])
assert np.array_equal(meta.id.to_numpy(str),ref_ids),'ID order differs from current v8; cannot combine'

# Save cumulative equity separately; reconstruct factors with per-year resets.
pth=W/'historical_7326_equity_2021Dec_2022_2023_2024.npz'
np.savez_compressed(pth,ids=ref_ids,equity_2021Dec=year_arrays['2021-12-01'],equity_2022=year_arrays['2022-01-01'],equity_2023=year_arrays['2023-01-01'],equity_2024=year_arrays['2024-01-01'])
print('SAVED',pth, 'MB',round(pth.stat().st_size/1e6,2),'elapsed',round(time.time()-t0,1),flush=True)
# Audit non-mutating: daily series different years; compare 2022 final with v7 legacy annual sweep, only inform not claim parity.
ref=pd.read_csv(P/'imba_fee006/imba_2022_tp_ttp_full_sweep_fee006.csv').drop_duplicates('strategy_id').set_index('strategy_id')
for name in [ref_ids[0],ref_ids[100],ref_ids[500],ref_ids[4157]]:
    i=np.where(ref_ids==name)[0][0]
    print('audit 2022',name,'new',round(100*(float(year_arrays['2022-01-01'][i,-1])-1),3),'prior2022',round(float(ref.loc[name,'return_2022_1x_pct']),3),flush=True)