#!/usr/bin/env python3
"""Sweep gate configs x TTP for a single pair. Writes results incrementally."""
import sys, os, json, time
proj = os.path.dirname(os.path.abspath(__file__)) + '/..'
os.chdir(proj)
sys.path.insert(0, os.path.join(proj, 'src'))
sys.path.insert(0, proj)
from scripts.backtest_multi_asset import run_backtest

pair = sys.argv[1]   # FETUSDT / XRPUSDT / BNBUSDT
box = float(sys.argv[2])
out_file = sys.argv[3]

ttp_range = [0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016]

GATES = {
    'FETUSDT': [
        dict(adx_on=27, adx_off=32, chop_on=0, chop_off=0, er_on=1.0, er_off=1.0, label='adx27/32'),
        dict(adx_on=27, adx_off=35, chop_on=0, chop_off=0, er_on=1.0, er_off=1.0, label='adx27/35'),
        dict(adx_on=30, adx_off=35, chop_on=0, chop_off=0, er_on=1.0, er_off=1.0, label='adx30/35'),
        dict(adx_on=30, adx_off=38, chop_on=0, chop_off=0, er_on=1.0, er_off=1.0, label='adx30/38'),
        dict(adx_on=33, adx_off=38, chop_on=0, chop_off=0, er_on=1.0, er_off=1.0, label='adx33/38'),
        dict(adx_on=33, adx_off=41, chop_on=0, chop_off=0, er_on=1.0, er_off=1.0, label='adx33/41'),
        dict(adx_on=36, adx_off=41, chop_on=0, chop_off=0, er_on=1.0, er_off=1.0, label='adx36/41'),
        dict(adx_on=42, adx_off=50, chop_on=30, chop_off=25, er_on=1.0, er_off=1.0, label='ch30+a42'),
        dict(adx_on=48, adx_off=56, chop_on=30, chop_off=25, er_on=1.0, er_off=1.0, label='ch30+a48'),
        dict(adx_on=54, adx_off=62, chop_on=30, chop_off=25, er_on=1.0, er_off=1.0, label='ch30+a54'),
    ],
    'XRPUSDT': [
        dict(chop_on=38, chop_off=33, adx_on=32, adx_off=42, er_on=0.25, er_off=0.40, label='30%a'),
        dict(chop_on=35, chop_off=30, adx_on=28, adx_off=34, er_on=0.40, er_off=0.50, label='33%'),
        dict(chop_on=38, chop_off=30, adx_on=28, adx_off=38, er_on=0.25, er_off=0.40, label='35%'),
        dict(chop_on=35, chop_off=30, adx_on=32, adx_off=38, er_on=0.25, er_off=0.35, label='38%'),
        dict(chop_on=35, chop_off=27, adx_on=32, adx_off=42, er_on=0.70, er_off=0.80, label='39%'),
        dict(chop_on=35, chop_off=30, adx_on=36, adx_off=46, er_on=0.35, er_off=0.50, label='42%'),
        dict(chop_on=35, chop_off=27, adx_on=40, adx_off=46, er_on=0.70, er_off=0.80, label='43%'),
        dict(chop_on=35, chop_off=27, adx_on=48, adx_off=54, er_on=0.30, er_off=0.40, label='45%'),
        dict(chop_on=38, chop_off=30, adx_on=44, adx_off=54, er_on=0.55, er_off=0.70, label='45%b'),
        dict(chop_on=38, chop_off=30, adx_on=48, adx_off=58, er_on=0.65, er_off=0.80, label='46%'),
    ],
    'BNBUSDT': [
        dict(chop_on=35, chop_off=30, adx_on=36, adx_off=42, er_on=0.25, er_off=0.40, label='30%'),
        dict(chop_on=38, chop_off=30, adx_on=36, adx_off=42, er_on=0.65, er_off=0.75, label='31%'),
        dict(chop_on=35, chop_off=30, adx_on=36, adx_off=46, er_on=0.70, er_off=0.80, label='32%'),
        dict(chop_on=35, chop_off=30, adx_on=44, adx_off=54, er_on=0.25, er_off=0.40, label='32b%'),
        dict(chop_on=38, chop_off=30, adx_on=48, adx_off=58, er_on=0.25, er_off=0.40, label='33%'),
        dict(chop_on=38, chop_off=30, adx_on=40, adx_off=50, er_on=0.50, er_off=0.65, label='33b%'),
        dict(chop_on=35, chop_off=30, adx_on=44, adx_off=50, er_on=0.65, er_off=0.80, label='34%'),
        dict(chop_on=35, chop_off=30, adx_on=44, adx_off=54, er_on=0.55, er_off=0.70, label='34b%'),
        dict(chop_on=35, chop_off=30, adx_on=48, adx_off=54, er_on=0.65, er_off=0.80, label='34c%'),
        dict(chop_on=38, chop_off=30, adx_on=48, adx_off=58, er_on=0.65, er_off=0.80, label='35%'),
    ],
}

gates = GATES[pair]
results = []
best = None; best_score = -9999
t0 = time.time()

for gi, gate in enumerate(gates):
    lbl = gate.pop('label')
    for ttp in ttp_range:
        r = run_backtest(pair, box=box, start_date='2024-01-01', quiet=True, fee_bps=20, ttp_trail_pct=ttp, **gate)
        c = r['combined']; f = r['flip_engine']; t = r['tp2_engine']
        score = c['total_return_pct'] + c['max_dd_pct'] * 0.5
        row = dict(gate=lbl, ttp=ttp, ret=c['total_return_pct'], dd=c['max_dd_pct'], trades=c['trades'],
                   wr=c['win_rate'], mean_pnl=c['mean_pnl_pct'], gon=r['gate_on_rate_pct'],
                   flip_ret=f['total_return_pct'], flip_tr=f['trades'], flip_wr=f['win_rate'],
                   tp2_ret=t['total_return_pct'], tp2_tr=t['trades'], tp2_wr=t['win_rate'], score=score)
        results.append(row)
        marker = ''
        if score > best_score:
            best_score = score; best = row; marker = ' ***'
        elapsed = time.time() - t0
        print(f'[{pair} {elapsed:>5.0f}s] {lbl:<10} ttp={ttp:.3f} | ret={c["total_return_pct"]:>7.1f}% dd={c["max_dd_pct"]:>6.1f}% tr={c["trades"]:>5} wr={c["win_rate"]:>5.1f}% gON={r["gate_on_rate_pct"]:>4.1f}% | flip={f["total_return_pct"]:>6.1f}%/{f["trades"]}tr tp2={t["total_return_pct"]:>7.1f}%/{t["trades"]}tr{marker}', flush=True)
    gate['label'] = lbl

# Save incrementally
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'\nBEST {pair}: gate={best["gate"]} ttp={best["ttp"]} ret={best["ret"]}% dd={best["dd"]}% trades={best["trades"]} wr={best["wr"]}%', flush=True)
print(f'Saved {len(results)} results to {out_file}', flush=True)
