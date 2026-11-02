"""Offline audit of TradingView Strategy Builder fills against visible indicator markers.

Time stamp decoding: imported Excel strings preserve Europe/Berlin *wall time* but
are incorrectly named entryUtc by the earlier importer. Treat them as naive local,
localize Europe/Berlin, then convert UTC; do not mutate source manifests.
No claim of independent entry/exit signal generation or order parity.
"""
from __future__ import annotations
import argparse,json,collections
from pathlib import Path
import numpy as np
import pandas as pd

def tv_as_local_berlin(iso):
    # ISO with appended Z in old importer is a labeling bug; remove artificial offset.
    return pd.Timestamp(iso).tz_localize(None).tz_localize('Europe/Berlin',ambiguous='raise',nonexistent='raise').tz_convert('UTC')

def run(manifest,binance_csv,kucoin_csv):
    m=json.loads(Path(manifest).read_text())
    datasets={}
    for key,f in [('BINANCE:SOLUSDT',binance_csv),('KUCOIN:SOLUSDT',kucoin_csv)]:
        d=pd.read_csv(f)
        d['timestamp_utc']=pd.to_datetime(d.time,unit='s',utc=True)
        d['second']=np.floor(d.time.to_numpy()).astype('int64')
        datasets[key]=d
    details=[];sums=[]
    for s in m['seeds']:
        sym=s['instrument']['symbol']
        if sym not in datasets:continue
        d=datasets[sym]
        time2=d['second'].to_numpy()
        counts=collections.Counter()
        candidates=s['trades']['closed']+s['trades']['open']
        for tr in candidates:
            side=tr['side'];key='Long Entry' if side=='long' else 'Short Entry'
            if side not in ('long','short'):continue
            t=(pd.Timestamp(tr['entryUtc']) if m.get('schemaVersion')==2 else tv_as_local_berlin(tr['entryUtc'])); sec=int(t.timestamp())
            rows=d.iloc[np.where(time2==sec)[0]]
            # Outside chart sample vs no Renko brick: distinct reasons.
            inside = int(time2.min()) <= sec <= int(time2.max())
            mark=bool((rows[key].fillna(0)!=0).any())
            anybar=len(rows)>0
            if inside:counts['within_time_range']+=1
            if anybar:counts['has_matching_renko_second']+=1
            if mark:counts['same_second_marker_same_direction']+=1
            if not anybar and inside:counts['no_exported_brick_at_this_second']+=1
            rsi_vals=pd.to_numeric(rows.get('30m',pd.Series([],dtype=float)),errors='coerce').dropna()
            rsi=float(rsi_vals.iloc[-1]) if len(rsi_vals) else None
            gate=bool(s['originalSettings']['safePropertyRows'][39][1]=='On') if side=='long' else bool(s['originalSettings']['safePropertyRows'][51][1]=='On')
            pass_rsi = (rsi>55 if side=='long' else rsi<75) if rsi is not None else None
            if mark and gate and pass_rsi is True:counts['marker_rsi_filter_pass']+=1
            if mark and gate and pass_rsi is False:counts['marker_rsi_filter_fail']+=1
            price_match=bool(rows.apply(lambda r: min(abs(tr['entryPrice']-r['open']),abs(tr['entryPrice']-r['close']))<0.011,axis=1).any()) if anybar else False
            if mark and price_match:counts['marker_and_price_same_brick_body']+=1
            details.append(dict(seed=s['id'],symbol=sym,tradeNumber=tr['number'],side=side,
               timestamp_as_imported=tr['entryUtc'],corrected_utc=t.isoformat(),inside_export_range=inside,
               renko_brick_second_exists=anybar,reference_marker_same_side=mark,entryPrice=tr['entryPrice'],
               entry_matches_renko_open_or_close=price_match,filter_enabled=gate,plot_30m_rsi=rsi,
               filter_pass_on_displayed_rsi=pass_rsi,reported_pnl=tr.get('referenceNetPnlUsdt')))
        sums.append(dict(seed=s['id'],symbol=sym,trades=len(candidates),**counts))
    return pd.DataFrame(details),pd.DataFrame(sums)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--manifest',default='/mnt/data/quant_tradingview_first_plants_reference_2026-11-02.json');a.add_argument('--binance-csv',default='/mnt/data/BINANCE_SOLUSDT, 5 (5)(1).csv');a.add_argument('--kucoin-csv',default='/mnt/data/KUCOIN_SOLUSDT, 5 (2)(1).csv');a.add_argument('--prefix',default='/mnt/data/tv_ew_trade_audit');z=a.parse_args()
    details,sums=run(z.manifest,z.binance_csv,z.kucoin_csv)
    details.to_csv(z.prefix+'_entries.csv',index=False);sums.to_csv(z.prefix+'_summary.csv',index=False)
    print(sums.fillna(0).to_string(index=False))
    print('KUCOIN signal+RSI gates:')
    ku=details[(details.symbol=='KUCOIN:SOLUSDT') & details.reference_marker_same_side]
    print(ku.groupby(['side','filter_pass_on_displayed_rsi'],dropna=False).size().to_string())
    print('sample first 8 confirmed')
    print(ku[['tradeNumber','side','timestamp_as_imported','corrected_utc','entryPrice','plot_30m_rsi','filter_pass_on_displayed_rsi']].head(8).to_string(index=False))