"""Market-only construction; exported chart and signals are evaluator targets only.

The historical resolution transition and observed 7/5 HTF lengths are explicit
discovered metadata, not assumed to be the source Pine defaults. Both source
default and observed configurations are evaluated. This is historical parity,
not yet a realtime causal trading system.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from independent_htf import aggregate_minutes, build_renko, htf_expression, map_expression, evaluate_raw, confirmed_htf_state
from ew_replay import Config, replay_export


def historical_chart(minutes, transition, bootstrap_hours=None):
    hourly=aggregate_minutes(minutes,3600)
    if bootstrap_hours is not None:
        hourly=pd.concat([bootstrap_hours.loc[bootstrap_hours.time<hourly.time.min()],hourly],ignore_index=True)
    hourly['available_ms']=(hourly.time+3600)*1000
    five=aggregate_minutes(minutes,300)
    five['available_ms']=(five.time+300)*1000
    bars=pd.concat([hourly.loc[hourly.time<transition],five.loc[five.time>=transition]],ignore_index=True)
    return build_renko(bars,300)


def market_input(minutes, chart, htf_specs, causal=False, bootstrap_hours=None):
    frame=chart.copy()
    coverage=[]
    for k,(seconds,length) in enumerate(htf_specs,1):
        bars=aggregate_minutes(minutes,seconds)
        if bootstrap_hours is not None:
            prior=aggregate_minutes(bootstrap_hours,seconds,input_seconds=3600)
            bars=pd.concat([prior.loc[prior.time<bars.time.min()],bars],ignore_index=True)
        renko=build_renko(bars,seconds)
        expression=htf_expression(renko,length)
        mapped=confirmed_htf_state(renko,length,chart.available_ms) if causal else map_expression(expression,chart.time_ms)
        pre=f'Audit htf{k} '
        # Prefix kept only as the existing numerical engine's input schema.
        # Every value here is computed from minutes, never read from audit CSV.
        frame[pre+'active']=1
        frame[pre+'new period']=np.r_[False,np.diff(chart.time_ms.to_numpy()//(seconds*1000))!=0].astype(int)
        for col in ['pivot high raw','pivot low raw','ATR','pivot time','pivot idx']:
            frame[pre+col]=mapped[col].to_numpy()
        frame[f'htf{k}_source_available_ms']=mapped.available_ms.to_numpy()
        coverage.append((int(bars.time.max())+seconds)*1000)
    return frame,min(coverage)


def signal_comparison(reference, predicted, coverage_end, start):
    coverage_end=min(coverage_end,int(reference.time_ms.max())+1)
    start=max(start,int(reference.time_ms.min()))
    ref=reference.loc[(reference.time_ms>=start)&(reference.time_ms<coverage_end)]
    pred=predicted.loc[(predicted.time_ms>=start)&(predicted.time_ms<coverage_end)]
    report={}
    for col in ['Long Entry','Short Entry','Wave 5 Exit (bullish impulse)','Wave 5 Exit (bearish impulse)']:
        a=set(ref.loc[ref[col]==1,'time_ms']);b=set(pred.loc[pred[col]==1,'time_ms'])
        report[col]={'reference':len(a),'candidate':len(b),'matched':len(a&b),'missing':sorted(a-b),'extra':sorted(b-a)}
    return report


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    minutes=pd.read_csv(root/'inputs/binance_spot_minutes.csv.gz')
    transition=int(pd.Timestamp('2026-09-07T00:00:00Z').timestamp())
    chart=historical_chart(minutes,transition)
    chart.to_csv(root/'results/market_only_chart.csv',index=False)
    # Freeze configurations before loading target reference series.
    configs=[('audit_observed_13_auto_7_5',13,[(3600,7),(14400,5)]),
             ('pine_defaults_13_auto_8_7',13,[(3600,8),(14400,7)]),
             ('original_11_auto_8_7',11,[(3600,8),(14400,7)]),
             ('original_11_manual_8_8',11,[(14400,8),(86400,8)])]
    predictions={}
    for name,primary,specs in configs:
        frame,end=market_input(minutes,chart,specs)
        pred=replay_export(frame,Config(primary=primary))
        pred.insert(0,'time_ms',chart.time_ms)
        pred.to_csv(root/f'results/{name}_signals.csv',index=False)
        predictions[name]=(pred,end)
        if name.startswith('audit_observed'):
            frame.to_csv(root/'results/market_only_inputs.csv',index=False)
    targets={}
    for suffix in [4,5]:
        ref=pd.read_csv(root/f'inputs/BINANCE_SOLUSDT, 5 ({suffix}).csv',float_precision='round_trip')
        ref['time_ms']=np.rint(ref.time*1000).astype('int64')
        targets[suffix]=ref
    reports={name:{'coverage_end_ms':end,'scores':{str(suffix):signal_comparison(ref,pred,end,transition*1000) for suffix,ref in targets.items()}}
             for name,(pred,end) in predictions.items()}
    (root/'results/market_only_signal_report.json').write_text(json.dumps(reports,indent=2)+'\n')
    for name,r in reports.items():
        print(name,'coverage_end',pd.to_datetime(r['coverage_end_ms'],unit='ms',utc=True))
        for suffix,scores in r['scores'].items():
            print(' export',suffix,{c:{k:v if k not in ['missing','extra'] else len(v) for k,v in s.items()} for c,s in scores.items()})
