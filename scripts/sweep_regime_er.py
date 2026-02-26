# scripts/sweep_regime_er.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from quant.strategies.signal_io import load_signals
from quant.strategies.flip_engine import FlipParams, run_flip_state_machine

# reuse regime + io helpers from runner (single source of truth)
from quant.backtest.renko_runner import _read_ohlcv_parquet, _signals_bundle_to_df, _build_regime_on


def _pair_trades_from_events(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=["entry_ts", "exit_ts", "side", "entry_px", "exit_px", "exit_event", "pnl_pct"])

    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, errors="coerce")
    ev = ev.dropna(subset=["ts"])
    sort_cols = ["ts", "seq"] if "seq" in ev.columns else ["ts"]
    ev = ev.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    exits = {"tp_exit", "sl_exit", "signal_flip_exit", "be_exit"}
    open_entry = None
    out = []
    for _, r in ev.iterrows():
        e = str(r.get("event", ""))
        if e == "entry":
            open_entry = r
            continue
        if e in exits and open_entry is not None:
            out.append(
                {
                    "entry_ts": pd.Timestamp(open_entry["ts"]),
                    "exit_ts": pd.Timestamp(r["ts"]),
                    "side": int(open_entry.get("side", 0)),
                    "entry_px": float(open_entry.get("price", np.nan)),
                    "exit_px": float(r.get("price", np.nan)),
                    "exit_event": e,
                    "pnl_pct": float(r.get("pnl_pct", np.nan)),
                }
            )
            open_entry = None
    return pd.DataFrame(out)


def _equity_from_trades(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame({"ts": [], "equity": []})

    t = trades.dropna(subset=["exit_ts", "pnl_pct"]).copy()
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True)
    t["pnl_pct"] = pd.to_numeric(t["pnl_pct"], errors="coerce")
    t = t.dropna(subset=["pnl_pct"]).sort_values("exit_ts").reset_index(drop=True)

    eq = float(initial_capital) * np.cumprod(1.0 + t["pnl_pct"].astype(float).values)
    return pd.DataFrame({"ts": t["exit_ts"].values, "equity": eq})


def _run_one(
    bars: pd.DataFrame,
    signals_df: pd.DataFrame,
    params: FlipParams,
    regime_on: pd.Series | None,
) -> Tuple[float, float, int]:
    _, events, _term = run_flip_state_machine(
        bars=bars[["ts", "open", "high", "low", "close"]].copy(),
        signals_df=signals_df,
        params=params,
        regime_on=regime_on,
    )

    trades = _pair_trades_from_events(events)
    equity = _equity_from_trades(trades, initial_capital=10_000.0)

    equity0 = 10_000.0
    equity1 = float(equity["equity"].iloc[-1]) if len(equity) else equity0
    total_return_pct = (equity1 / equity0 - 1.0) * 100.0

    if len(equity):
        peak = equity["equity"].cummax()
        dd = (equity["equity"] / peak - 1.0) * 100.0
        max_drawdown_pct = float(dd.min())
    else:
        max_drawdown_pct = 0.0

    return float(total_return_pct), float(max_drawdown_pct), int(len(trades))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--parquet", required=True)
    ap.add_argument("--signals-jsonl", required=True)
    ap.add_argument("--box", type=float, default=0.1)

    # fixed strategy params
    ap.add_argument("--fee-bps", type=float, default=3.0)
    ap.add_argument("--ttp-trail-pct", type=float, default=0.00775)
    ap.add_argument("--sl-cap-pct", type=float, default=0.011)
    ap.add_argument("--swing-lookback", type=int, default=50)
    ap.add_argument("--be-trigger-pct", type=float, default=0.0)
    ap.add_argument("--be-offset-pct", type=float, default=0.0)

    # regime choice
    ap.add_argument(
        "--regime",
        type=str,
        default="er",
        help="er | chop_er | adx_er | chop_adx_er | ... (uses renko_runner _build_regime_on)",
    )

    # keep CHOP/ADX params constant unless you sweep them separately
    ap.add_argument("--chop-len", type=int, default=14)
    ap.add_argument("--chop-on", type=float, default=54.0)
    ap.add_argument("--chop-off", type=float, default=48.0)
    ap.add_argument("--adx-len", type=int, default=14)
    ap.add_argument("--adx-on", type=float, default=18.0)
    ap.add_argument("--adx-off", type=float, default=25.0)

    # ER sweep grids
    ap.add_argument("--er-len", type=int, default=40)
    ap.add_argument("--er-on-grid", type=str, default="0.20,0.25,0.30,0.35")
    ap.add_argument("--er-off-grid", type=str, default="0.30,0.35,0.40,0.45")

    ap.add_argument("--out-dir", type=str, default=None)

    args = ap.parse_args()

    def _parse_floats(s: str) -> List[float]:
        return [float(x.strip()) for x in str(s).split(",") if x.strip()]

    er_on_grid = _parse_floats(args.er_on_grid)
    er_off_grid = _parse_floats(args.er_off_grid)

    bars = _read_ohlcv_parquet(args.parquet)
    bundle = load_signals(path=args.signals_jsonl, kind="jsonl")
    signals_df = _signals_bundle_to_df(bundle)

    params = FlipParams(
        fee_bps=float(args.fee_bps),
        ttp_trail_pct=float(args.ttp_trail_pct),
        sl_cap_pct=float(args.sl_cap_pct),
        swing_lookback=int(args.swing_lookback),
        be_trigger_pct=float(args.be_trigger_pct),
        be_offset_pct=float(args.be_offset_pct),
    )

    sweep_id = datetime.now(timezone.utc).strftime("er_sweep_%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (Path("data/sweeps") / sweep_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    total = 0

    for er_on in er_on_grid:
        for er_off in er_off_grid:
            if er_off <= er_on:
                continue

            regime_on = _build_regime_on(
                bars=bars[["ts", "open", "high", "low", "close"]].copy(),
                mode=str(args.regime),
                chop_len=int(args.chop_len),
                chop_on=float(args.chop_on),
                chop_off=float(args.chop_off),
                adx_len=int(args.adx_len),
                adx_on=float(args.adx_on),
                adx_off=float(args.adx_off),
                er_len=int(args.er_len),
                er_on=float(er_on),
                er_off=float(er_off),
            )

            on_rate = float(regime_on.mean()) * 100.0 if regime_on is not None else 100.0
            ret, dd, trades = _run_one(bars, signals_df, params, regime_on)

            total += 1
            print(
                f"[{total}] regime={args.regime} ER(len={args.er_len}) on={er_on:.3f} off={er_off:.3f} "
                f"ON-rate={on_rate:.2f}% ret={ret:.4f} dd={dd:.4f} trades={trades}"
            )

            rows.append(
                {
                    "regime": str(args.regime),
                    "er_len": int(args.er_len),
                    "er_on": float(er_on),
                    "er_off": float(er_off),
                    "on_rate_pct": float(on_rate),
                    "total_return_pct": float(ret),
                    "max_drawdown_pct": float(dd),
                    "trades": int(trades),
                    "fee_bps": float(args.fee_bps),
                    "ttp_trail_pct": float(args.ttp_trail_pct),
                    "sl_cap_pct": float(args.sl_cap_pct),
                    "swing_lookback": int(args.swing_lookback),
                    "be_trigger_pct": float(args.be_trigger_pct),
                    "be_offset_pct": float(args.be_offset_pct),
                    "parquet": str(args.parquet),
                    "signals_jsonl": str(args.signals_jsonl),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["total_return_pct", "max_drawdown_pct"], ascending=[False, True]).reset_index(drop=True)

    (out_dir / "results.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    df.to_parquet(out_dir / "results.parquet", index=False)

    top = df.head(30).copy()
    (out_dir / "top30.csv").write_text(top.to_csv(index=False), encoding="utf-8")

    meta = {
        "sweep_id": sweep_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "count": int(len(df)),
        "regime": str(args.regime),
        "er_len": int(args.er_len),
        "er_on_grid": er_on_grid,
        "er_off_grid": er_off_grid,
        "fixed_params": {
            "fee_bps": float(args.fee_bps),
            "ttp_trail_pct": float(args.ttp_trail_pct),
            "sl_cap_pct": float(args.sl_cap_pct),
            "swing_lookback": int(args.swing_lookback),
            "be_trigger_pct": float(args.be_trigger_pct),
            "be_offset_pct": float(args.be_offset_pct),
        },
        "chop_params": {"len": int(args.chop_len), "on": float(args.chop_on), "off": float(args.chop_off)},
        "adx_params": {"len": int(args.adx_len), "on": float(args.adx_on), "off": float(args.adx_off)},
        "inputs": {"parquet": str(args.parquet), "signals_jsonl": str(args.signals_jsonl), "box": float(args.box)},
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"WROTE: {out_dir / 'results.csv'}")
    print(f"WROTE: {out_dir / 'results.parquet'}")
    print(f"WROTE: {out_dir / 'top30.csv'}")
    print(f"WROTE: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
