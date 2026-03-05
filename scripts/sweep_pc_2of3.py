#!/usr/bin/env python3
"""Sweep PC 2-of-3 gate (drift/elasticity/instability) with quantile thresholds. ~15 runs per pair."""
import sys, os, json, time
proj = os.path.dirname(os.path.abspath(__file__)) + '/..'
os.chdir(proj)
sys.path.insert(0, os.path.join(proj, 'src'))
sys.path.insert(0, proj)
from scripts.backtest_multi_asset import run_backtest

pair = sys.argv[1]
box = float(sys.argv[2])
out_file = sys.argv[3]

# 5 quantile configs x 3 TTP values = 15 per pair
# The 3 quantile knobs: q_drift (drift threshold), q_elas (elasticity threshold), q_instab (instability threshold)
# Higher q_drift = more permissive drift (wider gate ON)
# Lower q_elas = more permissive elasticity (wider gate ON)
# Higher q_instab = more permissive instability (wider gate ON)

CONFIGS = {
    'BNBUSDT': {
        'ttp_range': [0.010, 0.011, 0.012],
        'gates': [
            # Tight gate (fewer ON bars): low q_drift, high q_elas, low q_instab
            dict(q_drift=0.50, q_elas=0.40, q_instab=0.35, drift_win=240, elas_h=15, label='tight'),
            # Default from gate_provider.py
            dict(q_drift=0.60, q_elas=0.30, q_instab=0.40, drift_win=240, elas_h=15, label='default'),
            # Loose gate (more ON bars)
            dict(q_drift=0.70, q_elas=0.20, q_instab=0.50, drift_win=240, elas_h=15, label='loose'),
            # Short drift window (faster reaction)
            dict(q_drift=0.60, q_elas=0.30, q_instab=0.40, drift_win=120, elas_h=15, label='fast_drift'),
            # Longer elasticity horizon
            dict(q_drift=0.60, q_elas=0.30, q_instab=0.40, drift_win=240, elas_h=30, label='long_elas'),
        ],
    },
    'XRPUSDT': {
        'ttp_range': [0.013, 0.014, 0.015],
        'gates': [
            dict(q_drift=0.50, q_elas=0.40, q_instab=0.35, drift_win=240, elas_h=15, label='tight'),
            dict(q_drift=0.60, q_elas=0.30, q_instab=0.40, drift_win=240, elas_h=15, label='default'),
            dict(q_drift=0.70, q_elas=0.20, q_instab=0.50, drift_win=240, elas_h=15, label='loose'),
            dict(q_drift=0.60, q_elas=0.30, q_instab=0.40, drift_win=120, elas_h=15, label='fast_drift'),
            dict(q_drift=0.60, q_elas=0.30, q_instab=0.40, drift_win=240, elas_h=30, label='long_elas'),
        ],
    },
    'FETUSDT': {
        'ttp_range': [0.007, 0.009, 0.011],
        'gates': [
            dict(q_drift=0.50, q_elas=0.40, q_instab=0.35, drift_win=240, elas_h=15, label='tight'),
            dict(q_drift=0.60, q_elas=0.30, q_instab=0.40, drift_win=240, elas_h=15, label='default'),
            dict(q_drift=0.70, q_elas=0.20, q_instab=0.50, drift_win=240, elas_h=15, label='loose'),
            dict(q_drift=0.60, q_elas=0.30, q_instab=0.40, drift_win=120, elas_h=15, label='fast_drift'),
            dict(q_drift=0.60, q_elas=0.30, q_instab=0.40, drift_win=240, elas_h=30, label='long_elas'),
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
        r = run_backtest(pair, box=box, start_date='2024-01-01', quiet=True, fee_bps=20,
                         regime_mode='pc_2of3', ttp_trail_pct=ttp, **gate)
        c = r['combined']; f = r['flip_engine']; t = r['tp2_engine']
        score = c['total_return_pct'] + c['max_dd_pct'] * 0.5
        row = dict(gate=lbl, ttp=ttp, ret=c['total_return_pct'], dd=c['max_dd_pct'], trades=c['trades'],
                   wr=c['win_rate'], mean_pnl=c['mean_pnl_pct'], gon=r['gate_on_rate_pct'],
                   flip_ret=f['total_return_pct'], flip_tr=f['trades'], flip_wr=f['win_rate'],
                   tp2_ret=t['total_return_pct'], tp2_tr=t['trades'], tp2_wr=t['win_rate'], score=score,
                   **{k: gate.get(k) for k in ['q_drift', 'q_elas', 'q_instab', 'drift_win', 'elas_h']})
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
