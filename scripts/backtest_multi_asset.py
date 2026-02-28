#!/usr/bin/env python3
"""
Multi-asset backtest: replicate SOL trading logic (Flip Engine + TP2) for BNB, FET, XRP.
Self-contained script that imports from the existing quant package.

Usage:
  python scripts/backtest_multi_asset.py --pair FETUSDT --box 0.0001 \
    --ttp-trail-pct 0.012 --min-sl-pct 0.015 --max-sl-pct 0.03 \
    --tp1-pct 0.04 --tp2-pct 0.08 --tp2-min-sl-pct 0.03 --tp2-max-sl-pct 0.08 \
    --regime chop_adx_er --run-id FET_test1
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quant.features.renko import renko_from_close
from quant.strategies.imba import ImbaParams, compute_imba_signals, write_signals_jsonl
from quant.strategies.flip_engine import FlipParams, run_flip_state_machine
from quant.backtest.renko_runner_tp2 import (
    TP2Params,
    _signals_to_brick_events,
    legs_to_trades,
    run_tp2_engine,
)


def load_csv_gz(pair: str, data_dir: str = "data", start_date: str = "2024-01-01") -> pd.DataFrame:
    path = os.path.join(data_dir, f"{pair}.csv.gz")
    df = pd.read_csv(
        path, sep="|", header=None,
        names=["ts", "open", "high", "low", "close", "v1", "v2", "v3", "v4", "trades"],
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if start_date:
        cutoff = pd.to_datetime(start_date, utc=True)
        df = df[df["ts"] >= cutoff].reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close"]]


# ---- Regime indicators (same as renko_runner.py) ----
def _true_range(df):
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    pc = c.shift(1)
    return pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def _choppiness(df, n):
    tr = _true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = df["high"].astype(float).rolling(n, min_periods=n).max()
    ll = df["low"].astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))

def _wilder_smooth(x, n):
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()

def _adx(df, n):
    h, l = df["high"].astype(float), df["low"].astype(float)
    up, down = h.diff(), -l.diff()
    dm_p = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_m = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = _true_range(df)
    atr = _wilder_smooth(tr, n)
    sp, sm = _wilder_smooth(dm_p, n), _wilder_smooth(dm_m, n)
    dip = 100.0 * (sp / atr.replace(0, np.nan))
    dim = 100.0 * (sm / atr.replace(0, np.nan))
    dx = 100.0 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return _wilder_smooth(dx, n)

def _efficiency_ratio(df, n):
    c = df["close"].astype(float)
    net = (c - c.shift(n)).abs()
    denom = c.diff().abs().rolling(n, min_periods=n).sum()
    return (net / denom.replace(0, np.nan)).clip(0, 1)

def _hysteresis_onoff(x, on_th, off_th):
    state = False
    out = []
    for v in x.values:
        if np.isnan(v):
            out.append(state); continue
        if not state and v >= on_th: state = True
        elif state and v <= off_th: state = False
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")

def _hysteresis_low(x, on_th, off_th, start_on=True):
    state = bool(start_on)
    out = []
    for v in x.values:
        if np.isnan(v):
            out.append(state); continue
        if state and v >= off_th: state = False
        elif (not state) and v <= on_th: state = True
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")


def build_regime(bars, chop_on=58, chop_off=52, adx_on=18, adx_off=25,
                 er_on=0.30, er_off=0.40, chop_len=14, adx_len=14, er_len=40):
    """Build Gate ON (countertrend) regime: CHOP high + ADX low + ER low."""
    df = bars.copy().reset_index(drop=True)
    chop = _choppiness(df, chop_len)
    chop_ok = _hysteresis_onoff(chop, chop_on, chop_off)
    adx_v = _adx(df, adx_len)
    adx_ok = _hysteresis_low(adx_v, adx_on, adx_off, start_on=True)
    er = _efficiency_ratio(df, er_len)
    er_ok = _hysteresis_low(er, er_on, er_off, start_on=True)
    regime = chop_ok & adx_ok & er_ok
    ts_idx = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
    regime.index = ts_idx
    return regime


def equity_stats(pnls):
    if len(pnls) == 0:
        return {"total_return_pct": 0, "max_dd_pct": 0, "trades": 0, "win_rate": 0, "mean_pnl_pct": 0}
    eq = np.cumprod(1.0 + np.array(pnls, dtype=float))
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    wins = sum(1 for p in pnls if p > 0)
    return {
        "total_return_pct": round((eq[-1] - 1.0) * 100, 2),
        "max_dd_pct": round(float(dd.min()) * 100, 2),
        "trades": len(pnls),
        "win_rate": round(wins / len(pnls) * 100, 1),
        "mean_pnl_pct": round(float(np.mean(pnls)) * 100, 3),
    }


def run_backtest(
    pair: str,
    box: float,
    # Flip (Gate ON) params
    ttp_trail_pct: float = 0.012,
    min_sl_pct: float = 0.015,
    max_sl_pct: float = 0.030,
    swing_lookback: int = 50,
    fee_bps: float = 4.0,
    # TP2 (Gate OFF) params
    tp1_pct: float = 0.04,
    tp2_pct: float = 0.08,
    tp1_frac: float = 0.5,
    tp2_min_sl_pct: float = 0.03,
    tp2_max_sl_pct: float = 0.08,
    tp2_swing_lookback: int = 50,
    flip_on_opposite: bool = True,
    # Regime params
    chop_on: float = 58.0,
    chop_off: float = 52.0,
    adx_on: float = 18.0,
    adx_off: float = 25.0,
    er_on: float = 0.30,
    er_off: float = 0.40,
    # IMBA params
    imba_lookback: int = 240,
    imba_sl_abs: float = None,  # auto-set based on box
    # Other
    run_id: str = None,
    data_dir: str = "data",
    quiet: bool = False,
    start_date: str = "2024-01-01",
) -> dict:
    t0 = time.time()

    # Load data
    ohlcv = load_csv_gz(pair, data_dir, start_date=start_date)
    if not quiet:
        print(f"[{pair}] Loaded {len(ohlcv)} 1m bars, {ohlcv.ts.iloc[0]} -> {ohlcv.ts.iloc[-1]}")

    # Build Renko
    bricks = renko_from_close(ohlcv, box=box)
    bricks["high"] = bricks[["open", "close"]].max(axis=1)
    bricks["low"] = bricks[["open", "close"]].min(axis=1)
    if not quiet:
        print(f"[{pair}] Renko box={box} -> {len(bricks)} bricks")

    # IMBA signals
    if imba_sl_abs is None:
        imba_sl_abs = box * 15  # scale SL with box
    imba_params = ImbaParams(lookback=imba_lookback, fixed_sl_abs=imba_sl_abs)
    signals_df = compute_imba_signals(bricks, imba_params)
    if not quiet:
        print(f"[{pair}] IMBA signals: {len(signals_df)}")

    # Build regime on bricks
    regime_on = build_regime(
        bricks, chop_on=chop_on, chop_off=chop_off,
        adx_on=adx_on, adx_off=adx_off, er_on=er_on, er_off=er_off,
    )
    gate_on_rate = float(regime_on.mean()) * 100
    gate_off = ~regime_on  # Gate OFF = inverse
    gate_off_rate = float(gate_off.mean()) * 100
    if not quiet:
        print(f"[{pair}] Regime: Gate ON rate={gate_on_rate:.1f}%, Gate OFF rate={gate_off_rate:.1f}%")

    # ========== Strategy 1: Flip Engine (Gate ON = countertrend) ==========
    flip_params = FlipParams(
        fee_bps=fee_bps,
        ttp_trail_pct=ttp_trail_pct,
        min_sl_pct=min_sl_pct,
        max_sl_pct=max_sl_pct,
        swing_lookback=swing_lookback,
    )

    _, flip_events, _ = run_flip_state_machine(
        bars=bricks[["ts", "open", "high", "low", "close"]].copy(),
        signals_df=signals_df[["ts", "signal"]].copy(),
        params=flip_params,
        regime_on=regime_on,
    )

    # Pair flip trades
    from quant.backtest.renko_runner import _pair_trades_from_events
    flip_trades = _pair_trades_from_events(flip_events, price_col="price")
    flip_pnls = flip_trades["pnl_pct"].astype(float).tolist() if len(flip_trades) else []
    flip_stats = equity_stats(flip_pnls)

    # ========== Strategy 2: TP2 Engine (Gate OFF = trendfollower) ==========
    sig_event = _signals_to_brick_events(bricks["ts"], signals_df[["ts", "signal"]])
    gate_off_series = pd.Series(gate_off.astype(int).values)

    tp2_params = TP2Params(
        tp1_pct=tp1_pct,
        tp2_pct=tp2_pct,
        tp1_frac=tp1_frac,
        min_sl_pct=tp2_min_sl_pct,
        max_sl_pct=tp2_max_sl_pct,
        swing_lookback=tp2_swing_lookback,
        flip_on_opposite=flip_on_opposite,
    )

    tp2_events, tp2_legs = run_tp2_engine(
        bricks=bricks[["ts", "open", "high", "low", "close"]].copy(),
        sig_event=sig_event,
        gate_on=gate_off_series,
        params=tp2_params,
    )
    tp2_trades = legs_to_trades(tp2_legs)
    tp2_pnls = tp2_trades["pnl_pct"].astype(float).tolist() if len(tp2_trades) else []
    tp2_stats = equity_stats(tp2_pnls)

    # ========== Combined ==========
    all_pnls = []
    combined_entries = []
    if len(flip_trades):
        for _, r in flip_trades.iterrows():
            combined_entries.append((r["exit_ts"], float(r["pnl_pct"]), "on"))
    if len(tp2_trades):
        for _, r in tp2_trades.iterrows():
            combined_entries.append((r["exit_ts"], float(r["pnl_pct"]), "off"))
    combined_entries.sort(key=lambda x: x[0])
    combined_pnls = [p for _, p, _ in combined_entries]
    combined_stats = equity_stats(combined_pnls)

    elapsed = time.time() - t0

    result = {
        "pair": pair,
        "box": box,
        "bricks": len(bricks),
        "imba_signals": len(signals_df),
        "gate_on_rate_pct": round(gate_on_rate, 1),
        "gate_off_rate_pct": round(gate_off_rate, 1),
        "flip_engine": {
            "params": {"ttp": ttp_trail_pct, "min_sl": min_sl_pct, "max_sl": max_sl_pct},
            **flip_stats,
        },
        "tp2_engine": {
            "params": {"tp1": tp1_pct, "tp2": tp2_pct, "tp1_frac": tp1_frac,
                       "min_sl": tp2_min_sl_pct, "max_sl": tp2_max_sl_pct},
            **tp2_stats,
        },
        "combined": combined_stats,
        "regime_params": {
            "chop_on": chop_on, "chop_off": chop_off,
            "adx_on": adx_on, "adx_off": adx_off,
            "er_on": er_on, "er_off": er_off,
        },
        "elapsed_s": round(elapsed, 1),
    }

    if not quiet:
        print(f"\n{'='*60}")
        print(f"[{pair}] RESULTS (box={box}, {elapsed:.1f}s)")
        print(f"  Flip Engine (Gate ON):  {flip_stats}")
        print(f"  TP2 Engine  (Gate OFF): {tp2_stats}")
        print(f"  Combined:               {combined_stats}")
        print(f"{'='*60}\n")

    # Save results
    if run_id:
        out_dir = Path(data_dir) / "runs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "stats.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


def sweep_boxes(pair, boxes, **kwargs):
    """Sweep box sizes and return results."""
    results = []
    for box in boxes:
        try:
            r = run_backtest(pair=pair, box=box, quiet=True, **kwargs)
            results.append(r)
            print(f"  box={box}: combined return={r['combined']['total_return_pct']}%, "
                  f"dd={r['combined']['max_dd_pct']}%, trades={r['combined']['trades']}")
        except Exception as e:
            print(f"  box={box}: FAILED - {e}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--box", type=float, required=True)
    ap.add_argument("--ttp-trail-pct", type=float, default=0.012)
    ap.add_argument("--min-sl-pct", type=float, default=0.015)
    ap.add_argument("--max-sl-pct", type=float, default=0.030)
    ap.add_argument("--tp1-pct", type=float, default=0.04)
    ap.add_argument("--tp2-pct", type=float, default=0.08)
    ap.add_argument("--tp1-frac", type=float, default=0.5)
    ap.add_argument("--tp2-min-sl-pct", type=float, default=0.03)
    ap.add_argument("--tp2-max-sl-pct", type=float, default=0.08)
    ap.add_argument("--fee-bps", type=float, default=4.0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)
    run_backtest(
        pair=args.pair, box=args.box,
        ttp_trail_pct=args.ttp_trail_pct,
        min_sl_pct=args.min_sl_pct, max_sl_pct=args.max_sl_pct,
        tp1_pct=args.tp1_pct, tp2_pct=args.tp2_pct, tp1_frac=args.tp1_frac,
        tp2_min_sl_pct=args.tp2_min_sl_pct, tp2_max_sl_pct=args.tp2_max_sl_pct,
        fee_bps=args.fee_bps, run_id=args.run_id, data_dir=args.data_dir,
    )
