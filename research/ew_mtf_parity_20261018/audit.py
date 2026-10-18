"""Freeze inputs and report reference-fed diagnostic parity, never profitability."""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import shutil
import numpy as np
import openpyxl
import pandas as pd
from ew_replay import Config, replay_export

ROOT=Path(__file__).resolve().parent
DOWNLOADS=Path('/Users/martinpeter/Downloads')
INPUTS=ROOT/'inputs'
OUT=ROOT/'results'
INPUTS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
files=[DOWNLOADS/'EW_v3_2_MTF_AUDIT.pine',
       DOWNLOADS/'BINANCE_SOLUSDT, 5 (4).csv', DOWNLOADS/'BINANCE_SOLUSDT, 5 (5).csv',
       *sorted(DOWNLOADS.glob('Strategy_Builder*BINANCE*.xlsx'))]
manifest=[]
for source in files:
    target=INPUTS/source.name
    if target.exists() and target.read_bytes()!=source.read_bytes():
        raise RuntimeError(f'Frozen input changed: {target}')
    if not target.exists(): shutil.copy2(source,target)
    manifest.append({'path':str(target.relative_to(ROOT)),'source':str(source),
                     'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

frames={}
for suffix in [4,5]:
    path=INPUTS/f'BINANCE_SOLUSDT, 5 ({suffix}).csv'
    frame=pd.read_csv(path,float_precision='round_trip',dtype={'time':str})
    frame['time_ms']=[int(Decimal(v)*1000) for v in frame.time]
    assert frame.time_ms.is_monotonic_increasing and frame.time_ms.is_unique
    frames[suffix]=frame
frame=frames[5]
candidate=replay_export(frame,Config(primary=13))
candidate.insert(0,'time_ms',frame.time_ms)
candidate.to_csv(OUT/'diagnostic_replay.csv',index=False)
comparisons=[]; mismatches=[]
# Whole-export and fixed trade-window reports; no fitted warm-up cutoff.
trade_start=int(pd.Timestamp('2026-09-07T00:00:00Z').timestamp()*1000)
trade_mask=frame.time_ms>=trade_start
for col in candidate.columns:
    if col=='time_ms': continue
    ok=np.isclose(candidate[col],frame[col],rtol=0,atol=1e-10)
    ids=np.flatnonzero(~ok)
    comparisons.append({'field':col,'rows':len(frame),'mismatches':len(ids),
                        'last_mismatch_row':int(ids[-1]) if len(ids) else None,
                        'trade_window_mismatches':int((~ok & trade_mask).sum())})
    for i in ids:
        mismatches.append({'row':int(i),'time_ms':int(frame.time_ms.iloc[i]),'field':col,
                           'reference':float(frame[col].iloc[i]),'candidate':float(candidate[col].iloc[i])})
pd.DataFrame(comparisons).to_csv(OUT/'field_comparison.csv',index=False)
pd.DataFrame(mismatches).to_csv(OUT/'mismatches.csv',index=False)

signal_scores={}
for label,mask in [('full',np.ones(len(frame),bool)),('trade_window',trade_mask)]:
    signal_scores[label]={}
    for col in ['Long Entry','Short Entry','Wave 5 Exit (bullish impulse)','Wave 5 Exit (bearish impulse)']:
        ref=frame[col].astype(bool)&mask; pred=candidate[col].astype(bool)&mask
        signal_scores[label][col]={'reference':int(ref.sum()),'matched':int((ref&pred).sum()),
                                   'missing':int((ref&~pred).sum()),'extra':int((~ref&pred).sum())}

tf_results={}
t=frame.time_ms.to_numpy()//1000
for k in [1,2]:
    actual=frame[f'Audit htf{k} new period'].astype(bool).to_numpy()
    tf_results[str(k)]={}
    for hours in [1,4,24]:
        periods=t//(hours*3600); new=np.r_[False,np.diff(periods)!=0]
        tf_results[str(k)][str(hours)]=int((actual[1:]!=new[1:]).sum())

strategies=[]
for p in sorted(INPUTS.glob('*.xlsx')):
    book=openpyxl.load_workbook(p,read_only=True,data_only=True)
    props=list(book['Properties'].values)[1:]
    rows=list(book['Trades'].values); trades=pd.DataFrame(rows[1:],columns=rows[0])
    # Excel stores timezone-naive wall time. Keep UTC offset unproven until event matching.
    trades.to_csv(OUT/(p.stem+'_events.csv'),index=False)
    (OUT/(p.stem+'_properties.json')).write_text(json.dumps(props,ensure_ascii=False,indent=2,default=str)+'\n')
    strategies.append({'file':p.name,'events':len(trades),'entries':int(trades.Type.str.startswith('Entry').sum()),
                       'exits':int(trades.Type.str.startswith('Exit').sum()),
                       'open_events':int((trades.Signal=='Open').sum()),
                       'long_condition':props[16][1],'short_condition':props[17][1]})

summary={'mode':'reference_fed_htf_diagnostic','benchmark_verdict':'not_comparable',
         'reason':'Independent HTF construction, Strategy Builder execution and real fills not validated.',
         'config':vars(Config(primary=13)),'rows':len(frame),'signal_scores':signal_scores,
         'htf_period_mismatches_excluding_first_row':tf_results,
         'htf1_pivot_count_offset':sorted((frame['Audit htf1 pivot count']-candidate['Audit htf1 pivot count']).unique().tolist()),
         'source_signal_totals':{str(k):{c:int(v[c].sum()) for c in ['Long Entry','Short Entry']} for k,v in frames.items()},
         'strategies':strategies}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
