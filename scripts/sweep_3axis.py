#!/usr/bin/env python3
"""Coarse 3-axis gate sweep (CHOP+ADX+ER all active). ~15 runs per pair."""
import sys, os, json, time
proj = os.path.dirname(os.path.abspath(__file__)) + '/..'
os.chdir(proj)
sys.path.insert(0, os.path.join(proj, 'src'))
sys.path.insert(0, proj)
from scripts.backtest_multi_asset import run_backtest

pair = sys.argv[1]
box = float(sys.argv[2])
out_file = sys.argv[3]

# 5 gate configs x 3 TTP values = 15 per pair
# All gates have CHOP + ADX + ER active (3-axis)
# TTP ranges narrowed to each pair's sweet spot from round 1

CONFIGS = {
    'BNBUSDT': {
        'ttp_range': [0.010, 0.011, 0.012],
        'gates': [
            # Varying CHOP tightness with moderate ADX and ER
            dict(chop_on=35, chop_off=30, adx_on=34, adx_off=40, er_on=0.30, er_off=0.45, label='c35a34e30'),
            dict(chop_on=38, chop_off=32, adx_on=34, adx_off=40, er_on=0.30, er_off=0.45, label='c38a34e30'),
            # Tighter ER with looser ADX
            dict(chop_on=35, chop_off=30, adx_on=38, adx_off=46, er_on=0.20, er_off=0.35, label='c35a38e20'),
            # Looser ER with tighter ADX
            dict(chop_on=35, chop_off=30, adx_on=32, adx_off=38, er_on=0.45, er_off=0.60, label='c35a32e45'),
            # Wide CHOP + moderate everything
            dict(chop_on=40, chop_off=32, adx_on=36, adx_off=44, er_on=0.35, er_off=0.50, label='c40a36e35'),
        ],
    },
    'XRPUSDT': {
        'ttp_range': [0.013, 0.014, 0.015],
        'gates': [
            # Around the winning 30%a config but varying axes
            dict(chop_on=38, chop_off=33, adx_on=30, adx_off=40, er_on=0.20, er_off=0.35, label='c38a30e20'),
            dict(chop_on=38, chop_off=33, adx_on=34, adx_off=44, er_on=0.20, er_off=0.35, label='c38a34e20'),
            # Tighter CHOP
            dict(chop_on=36, chop_off=31, adx_on=32, adx_off=42, er_on=0.25, er_off=0.40, label='c36a32e25'),
            # Looser ER to let more through
            dict(chop_on=38, chop_off=33, adx_on=32, adx_off=42, er_on=0.35, er_off=0.50, label='c38a32e35'),
            # Wider CHOP band
            dict(chop_on=40, chop_off=33, adx_on=32, adx_off=42, er_on=0.25, er_off=0.40, label='c40a32e25'),
        ],
    },
    'FETUSDT': {
        'ttp_range': [0.007, 0.009, 0.011],
        'gates': [
            # FET needs very loose CHOP/ER to get any ON time with 3 axes
            # Try loose CHOP (25-30) so it doesn't filter too aggressively
            dict(chop_on=25, chop_off=20, adx_on=28, adx_off=34, er_on=0.55, er_off=0.70, label='c25a28e55'),
            dict(chop_on=28, chop_off=22, adx_on=28, adx_off=34, er_on=0.55, er_off=0.70, label='c28a28e55'),
            dict(chop_on=25, chop_off=20, adx_on=30, adx_off=36, er_on=0.45, er_off=0.60, label='c25a30e45'),
            # Very loose CHOP + moderate ADX + tight ER
            dict(chop_on=30, chop_off=24, adx_on=28, adx_off=34, er_on=0.35, er_off=0.50, label='c30a28e35'),
            # Moderate all three
            dict(chop_on=28, chop_off=22, adx_on=32, adx_off=38, er_on=0.45, er_off=0.60, label='c28a32e45'),
        ],
    },
}

cfg = CONFIGS[pair]
gates = cfg['gates']
ttp_range = cfg['ttp_range']

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
        print(f'[{pair} {elapsed:>5.0f}s] {lbl:<12} ttp={ttp:.3f} | ret={c["total_return_pct"]:>7.1f}% dd={c["max_dd_pct"]:>6.1f}% tr={c["trades"]:>5} wr={c["win_rate"]:>5.1f}% gON={r["gate_on_rate_pct"]:>4.1f}% | flip={f["total_return_pct"]:>6.1f}%/{f["trades"]}tr tp2={t["total_return_pct"]:>7.1f}%/{t["trades"]}tr{marker}', flush=True)
    gate['label'] = lbl

    # Save after each gate config
    with open(out_file, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)

print(f'\nBEST {pair}: gate={best["gate"]} ttp={best["ttp"]} ret={best["ret"]}% dd={best["dd"]}% trades={best["trades"]} wr={best["wr"]}%', flush=True)
print(f'Saved {len(results)} results to {out_file}', flush=True)
