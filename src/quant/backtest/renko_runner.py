# src/quant/backtest/renko_runner.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from quant.strategies.signal_io import load_signals
from quant.strategies.flip_engine import FlipParams, run_flip_state_machine

# TP2 engine (Gate OFF strategy) – used when --regime-csv-off is provided
from quant.backtest.renko_runner_tp2 import (
    TP2Params,
    _signals_to_brick_events,
    legs_to_trades,
    run_tp2_engine,
)


def _read_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    need = {"open", "high", "low", "close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing columns: {sorted(missing)}")

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def _read_fills_parquet(path: str, fill_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("fills parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    if fill_col not in df.columns:
        raise ValueError(f"fills parquet missing fill_col='{fill_col}' (cols={list(df.columns)})")

    df[fill_col] = pd.to_numeric(df[fill_col], errors="coerce")
    df = df.dropna(subset=[fill_col]).reset_index(drop=True)

    return df[["ts", fill_col]].copy()


def _signals_bundle_to_df(bundle) -> pd.DataFrame:
    if hasattr(bundle, "signal") and bundle.signal is not None:
        s = bundle.signal.copy()
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
        s = s[~s.index.isna()]
        s = s[~s.index.duplicated(keep="last")]
        vals = pd.to_numeric(pd.Series(s.values), errors="coerce").fillna(0).astype(int).values
        sig_df = pd.DataFrame({"ts": s.index, "signal": vals})
        sig_df = sig_df[sig_df["signal"] != 0].copy()
        return sig_df.sort_values("ts").reset_index(drop=True)

    if hasattr(bundle, "df") and bundle.df is not None:
        df = bundle.df.copy()
        if "ts" not in df.columns:
            raise ValueError("signals bundle.df missing 'ts'")

        if "signal" not in df.columns and "position" in df.columns:
            df = df.rename(columns={"position": "signal"})
        if "signal" not in df.columns and "action" in df.columns:

            def _map(a):
                if a is None:
                    return 0
                a = str(a).strip().lower()
                if a in ("long", "buy", "1", "+1"):
                    return 1
                if a in ("short", "sell", "-1"):
                    return -1
                return 0

            df["signal"] = df["action"].map(_map)

        if "signal" not in df.columns:
            raise ValueError("signals bundle.df missing 'signal' (or 'position'/'action')")

        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int)
        df = df[df["signal"] != 0].copy()
        return df[["ts", "signal"]].reset_index(drop=True)

    raise ValueError("Unsupported signals bundle shape from load_signals()")


def _pair_trades_from_events(events: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """
    Pair trades for a CONTINUOUS-FLIP engine.

    Rules:
      - 'entry' opens a trade.
      - Any exit in {tp_exit, sl_exit, signal_flip_exit, be_exit, regime_exit} closes the current trade.
      - If the exit is a FLIP exit (tp_exit or signal_flip_exit), we immediately open a NEW trade
        in the opposite direction at the SAME timestamp/price (synthetic entry).
      - If the exit is a FLAT exit (sl_exit, be_exit, regime_exit), we go flat (no synthetic entry).
    """
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=["entry_ts", "exit_ts", "entry_px", "exit_px", "side", "exit_event", "pnl_pct"])

    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, errors="coerce")
    ev = ev.dropna(subset=["ts"])
    sort_cols = ["ts", "seq"] if "seq" in ev.columns else ["ts"]
    ev = ev.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    exits = {"tp_exit", "sl_exit", "signal_flip_exit", "be_exit", "regime_exit"}
    flip_exits = {"tp_exit", "signal_flip_exit"}
    flat_exits = {"sl_exit", "be_exit", "regime_exit"}

    open_side = None
    open_ts = None
    open_px = None

    out = []

    for _, r in ev.iterrows():
        e = str(r.get("event", ""))
        ts = pd.Timestamp(r["ts"])
        px = float(r.get(price_col, np.nan))
        side = int(r.get("side", 0))

        if e == "entry":
            open_side = side
            open_ts = ts
            open_px = px
            continue

        if e in exits and open_side is not None:
            out.append(
                {
                    "entry_ts": open_ts,
                    "exit_ts": ts,
                    "side": int(open_side),
                    "entry_px": float(open_px),
                    "exit_px": float(px),
                    "exit_event": e,
                    "pnl_pct": float(r.get("pnl_pct", np.nan)),
                }
            )

            if e in flip_exits:
                open_side = -int(open_side)
                open_ts = ts
                open_px = px
            elif e in flat_exits:
                open_side = None
                open_ts = None
                open_px = None

    return pd.DataFrame(out)


def _equity_from_trades(trades: pd.DataFrame, initial_capital: float = 1.0) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame({"ts": [], "equity": []})

    t = trades.copy()
    if "pnl_pct" not in t.columns:
        return pd.DataFrame({"ts": [], "equity": []})

    pnl_col = t["pnl_pct"]
    if isinstance(pnl_col, pd.DataFrame):
        pnl_series = pnl_col.iloc[:, -1]
    else:
        pnl_series = pnl_col

    t = t.copy()
    t["pnl_pct__tmp"] = pnl_series
    t = t.dropna(subset=["exit_ts", "pnl_pct__tmp"]).copy()
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True, errors="coerce")
    t = t.dropna(subset=["exit_ts"]).copy()

    t["pnl_pct__tmp"] = pd.to_numeric(t["pnl_pct__tmp"], errors="coerce")
    t = t.dropna(subset=["pnl_pct__tmp"]).sort_values("exit_ts").reset_index(drop=True)

    eq = float(initial_capital) * np.cumprod(1.0 + t["pnl_pct__tmp"].astype(float).values)
    return pd.DataFrame({"ts": t["exit_ts"].values, "equity": eq})


def _map_events_to_fills_asof(events: pd.DataFrame, fills: pd.DataFrame, fill_col: str) -> Tuple[pd.DataFrame, float]:
    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, errors="coerce")
    ev = ev.dropna(subset=["ts"]).sort_values(["ts", "seq"] if "seq" in ev.columns else ["ts"]).reset_index(drop=True)

    f = fills.copy()
    f["ts"] = pd.to_datetime(f["ts"], utc=True, errors="coerce")
    f = f.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    merged = pd.merge_asof(
        ev,
        f.rename(columns={fill_col: "price_real"}),
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    miss_rate = float(merged["price_real"].isna().mean()) if len(merged) else 0.0
    return merged, miss_rate


def _compute_trade_pnl_from_prices(trades: pd.DataFrame, fee_bps: float, price_cols: Tuple[str, str]) -> pd.DataFrame:
    t = trades.copy()
    e_col, x_col = price_cols
    t[e_col] = pd.to_numeric(t[e_col], errors="coerce")
    t[x_col] = pd.to_numeric(t[x_col], errors="coerce")
    t["side"] = pd.to_numeric(t["side"], errors="coerce")

    # fee_bps is ROUNDTRIP
    fee_rt = float(fee_bps) / 10000.0
    denom = t[e_col].abs().replace(0.0, np.nan)
    gross = (t["side"].astype(float) * (t[x_col].astype(float) - t[e_col].astype(float))) / denom
    t["pnl_pct_real"] = (gross - fee_rt).astype(float)
    return t


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def _choppiness(df: pd.DataFrame, n: int) -> pd.Series:
    n = int(n)
    tr = _true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = df["high"].astype(float).rolling(n, min_periods=n).max()
    ll = df["low"].astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    chop = 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))
    return chop


def _wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()


def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    n = int(n)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up = high.diff()
    down = -low.diff()

    dm_plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = _true_range(df)
    atr = _wilder_smooth(tr, n)
    sm_plus = _wilder_smooth(dm_plus, n)
    sm_minus = _wilder_smooth(dm_minus, n)

    di_plus = 100.0 * (sm_plus / atr.replace(0.0, np.nan))
    di_minus = 100.0 * (sm_minus / atr.replace(0.0, np.nan))

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan)
    adx = _wilder_smooth(dx, n)
    return adx


def _efficiency_ratio(df: pd.DataFrame, n: int) -> pd.Series:
    n = int(n)
    close = df["close"].astype(float)
    net = (close - close.shift(n)).abs()
    denom = close.diff().abs().rolling(n, min_periods=n).sum()
    er = net / denom.replace(0.0, np.nan)
    return er.clip(lower=0.0, upper=1.0)


def _hysteresis_onoff(x: pd.Series, on_th: float, off_th: float) -> pd.Series:
    on_th = float(on_th)
    off_th = float(off_th)
    state = False
    out = []
    for v in x.values:
        if np.isnan(v):
            out.append(state)
            continue
        if not state and v >= on_th:
            state = True
        elif state and v <= off_th:
            state = False
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")


def _hysteresis_low_is_good(x: pd.Series, on_th: float, off_th: float, start_on: bool = True) -> pd.Series:
    on_th = float(on_th)
    off_th = float(off_th)
    state = bool(start_on)
    out = []
    for v in x.values:
        if np.isnan(v):
            out.append(state)
            continue
        if state and v >= off_th:
            state = False
        elif (not state) and v <= on_th:
            state = True
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")


def _build_regime_on(
    bars: pd.DataFrame,
    mode: str,
    chop_len: int,
    chop_on: float,
    chop_off: float,
    adx_len: int,
    adx_on: float,
    adx_off: float,
    er_len: int,
    er_on: float,
    er_off: float,
) -> Optional[pd.Series]:
    mode = str(mode).strip().lower()
    if mode in ("none", "off", ""):
        return None

    df = bars.copy().reset_index(drop=True)
    parts = []

    if "chop" in mode:
        chop = _choppiness(df, int(chop_len))
        chop_ok = _hysteresis_onoff(chop, on_th=float(chop_on), off_th=float(chop_off))
        parts.append(chop_ok)

    if "adx" in mode:
        adx = _adx(df, int(adx_len))
        adx_ok = _hysteresis_low_is_good(adx, on_th=float(adx_on), off_th=float(adx_off), start_on=True)
        parts.append(adx_ok)

    if "er" in mode:
        er = _efficiency_ratio(df, int(er_len))
        er_ok = _hysteresis_low_is_good(er, on_th=float(er_on), off_th=float(er_off), start_on=True)
        parts.append(er_ok)

    if not parts:
        return None

    regime = parts[0]
    for p in parts[1:]:
        regime = regime & p

    ts_index = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True, errors="coerce"))
    regime.index = ts_index
    regime = regime[~regime.index.isna()]
    regime = regime[~regime.index.duplicated(keep="last")]
    return regime


def _load_external_regime_to_bricks(
    bars_ts: pd.Series,
    regime_csv: str,
    ts_col: str = "ts",
    value_col: str = "gate_on",
    default_off: bool = True,
) -> pd.Series:
    b = pd.DataFrame({"ts": pd.to_datetime(bars_ts, utc=True, errors="coerce")}).dropna().sort_values("ts")
    r = pd.read_csv(regime_csv)
    if ts_col not in r.columns:
        raise ValueError(f"regime csv missing ts_col='{ts_col}' (cols={list(r.columns)})")
    if value_col not in r.columns:
        raise ValueError(f"regime csv missing value_col='{value_col}' (cols={list(r.columns)})")

    r = r[[ts_col, value_col]].copy()
    r[ts_col] = pd.to_datetime(r[ts_col], utc=True, errors="coerce")
    r = r.dropna(subset=[ts_col]).sort_values(ts_col)
    r = r.drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)

    v = pd.to_numeric(r[value_col], errors="coerce")
    if v.isna().all():
        vv = r[value_col].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
        v = vv
    r["gate"] = v.fillna(0).astype(int).clip(0, 1)

    merged = pd.merge_asof(
        b,
        r[[ts_col, "gate"]].rename(columns={ts_col: "ts"}),
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )

    if default_off:
        merged["gate"] = merged["gate"].fillna(0).astype(int)
    else:
        merged["gate"] = merged["gate"].fillna(1).astype(int)

    out = pd.Series(merged["gate"].astype(bool).values, index=pd.DatetimeIndex(merged["ts"]), dtype="bool")
    out = out[~out.index.duplicated(keep="last")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--parquet", required=True)
    ap.add_argument("--box", type=float, default=0.1)
    ap.add_argument("--fee-bps", type=float, default=4.0)
    ap.add_argument("--signals-jsonl", required=True)
    ap.add_argument("--run-id", default=None)

    ap.add_argument("--ttp-trail-pct", type=float, default=0.012)

    ap.add_argument("--min-sl-pct", type=float, default=0.015)
    ap.add_argument("--max-sl-pct", type=float, default=0.030)

    ap.add_argument("--swing-lookback", type=int, default=250)

    ap.add_argument(
        "--regime",
        type=str,
        default="none",
        help="none | chop | adx | er | chop_adx | chop_er | adx_er | chop_adx_er",
    )

    ap.add_argument("--regime-csv", type=str, default=None)
    ap.add_argument("--regime-ts-col", type=str, default="ts")
    ap.add_argument("--regime-col", type=str, default="gate_on")
    ap.add_argument("--regime-default-off", action="store_true")

    # Gate OFF (TP2 strategy): use with --regime-csv-off; optional second gate for merged ON+OFF
    ap.add_argument("--regime-csv-off", type=str, default=None, help="Gate for OFF sessions (TP2 strategy). With --regime-csv: merged ON+OFF run.")
    ap.add_argument("--regime-ts-col-off", type=str, default="ts")
    ap.add_argument("--regime-col-off", type=str, default="gate_on")

    # TP2 params (Gate OFF strategy)
    ap.add_argument("--tp1-pct", type=float, default=0.015)
    ap.add_argument("--tp2-pct", type=float, default=0.030)
    ap.add_argument("--tp1-frac", type=float, default=0.5)
    ap.add_argument("--tp2-min-sl-pct", type=float, default=0.030)
    ap.add_argument("--tp2-max-sl-pct", type=float, default=0.080)
    ap.add_argument("--tp2-swing-lookback", type=int, default=180)
    ap.add_argument("--no-flip-on-opposite", action="store_true", help="TP2: do not flip on opposite signal")

    ap.add_argument("--chop-len", type=int, default=14)
    ap.add_argument("--chop-on", type=float, default=58.0)
    ap.add_argument("--chop-off", type=float, default=52.0)

    ap.add_argument("--adx-len", type=int, default=14)
    ap.add_argument("--adx-on", type=float, default=18.0)
    ap.add_argument("--adx-off", type=float, default=25.0)

    ap.add_argument("--er-len", type=int, default=40)
    ap.add_argument("--er-on", type=float, default=0.30)
    ap.add_argument("--er-off", type=float, default=0.40)

    ap.add_argument(
        "--fills-parquet",
        type=str,
        default=None,
        help="OHLC or fills parquet: use real prices for PnL (merge_asof on event ts). Writes trades_real.parquet + equity_real.parquet.",
    )
    ap.add_argument(
        "--fill-col",
        type=str,
        default="close",
        help="Column in fills-parquet for price (e.g. 'close' for OHLC close). Used only if --fills-parquet is set.",
    )

    args = ap.parse_args()

    bars = _read_ohlcv_parquet(args.parquet)
    print(f"INFO renko box={args.box} bricks={len(bars)}")
    print(f"INFO signals jsonl={args.signals_jsonl}")

    bundle = load_signals(path=args.signals_jsonl, kind="jsonl")
    signals_df = _signals_bundle_to_df(bundle)

    regime_on: Optional[pd.Series] = None
    if args.regime_csv:
        regime_on = _load_external_regime_to_bricks(
            bars_ts=bars["ts"],
            regime_csv=str(args.regime_csv),
            ts_col=str(args.regime_ts_col),
            value_col=str(args.regime_col),
            default_off=bool(args.regime_default_off),
        )
        on_rate = float(regime_on.mean()) * 100.0 if len(regime_on) else 0.0
        print(f"INFO regime=external csv={args.regime_csv} ON-rate={on_rate:.2f}%")
    else:
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
            er_on=float(args.er_on),
            er_off=float(args.er_off),
        )
        if regime_on is not None:
            on_rate = float(regime_on.mean()) * 100.0
            print(f"INFO regime={args.regime} ON-rate={on_rate:.2f}%")

    # Gate OFF (TP2 strategy): load when --regime-csv-off is set
    gate_off_aligned: Optional[pd.Series] = None
    if args.regime_csv_off:
        gate_off_aligned = _load_external_regime_to_bricks(
            bars_ts=bars["ts"],
            regime_csv=str(args.regime_csv_off),
            ts_col=str(args.regime_ts_col_off),
            value_col=str(args.regime_col_off),
            default_off=True,
        )
        off_rate = float(gate_off_aligned.mean()) * 100.0 if len(gate_off_aligned) else 0.0
        print(f"INFO regime OFF csv={args.regime_csv_off} ON-rate={off_rate:.2f}%")

    # Merged: flip during Gate ON, TP2 during Gate OFF. OFF-only: only TP2 when no ON gate.
    run_flip = (args.regime_csv is not None or args.regime != "none") or not args.regime_csv_off
    run_tp2 = bool(args.regime_csv_off)

    params = FlipParams(
        fee_bps=float(args.fee_bps),
        ttp_trail_pct=float(args.ttp_trail_pct),
        min_sl_pct=float(args.min_sl_pct),
        max_sl_pct=float(args.max_sl_pct),
        swing_lookback=int(args.swing_lookback),
    )

    events = None
    trades = pd.DataFrame()
    equity = pd.DataFrame({"ts": [], "equity": []})

    if run_flip:
        pos, events, *_ = run_flip_state_machine(
            bars=bars[["ts", "open", "high", "low", "close"]].copy(),
            signals_df=signals_df,
            params=params,
            regime_on=regime_on,
        )
        trades = _pair_trades_from_events(events, price_col="price")
        equity = _equity_from_trades(trades, initial_capital=1.0)

    events_tp2 = None
    legs_tp2 = None
    trades_tp2 = pd.DataFrame()
    equity_tp2 = pd.DataFrame({"ts": [], "equity": []})

    if run_tp2 and gate_off_aligned is not None:
        sig_event = _signals_to_brick_events(bricks_ts=bars["ts"], sig_df=signals_df)
        gate_on_tp2 = pd.Series(gate_off_aligned.astype(int).values)
        params_tp2 = TP2Params(
            tp1_pct=float(args.tp1_pct),
            tp2_pct=float(args.tp2_pct),
            tp1_frac=float(args.tp1_frac),
            min_sl_pct=float(args.tp2_min_sl_pct),
            max_sl_pct=float(args.tp2_max_sl_pct),
            swing_lookback=int(args.tp2_swing_lookback),
            flip_on_opposite=not args.no_flip_on_opposite,
        )
        events_tp2, legs_tp2 = run_tp2_engine(
            bricks=bars[["ts", "open", "high", "low", "close"]].copy(),
            sig_event=sig_event,
            gate_on=gate_on_tp2,
            params=params_tp2,
        )
        trades_tp2 = legs_to_trades(legs_tp2)
        if len(trades_tp2):
            pnl = trades_tp2["pnl_pct"].astype(float)
            eq = float(1.0) * np.cumprod(1.0 + pnl.values)
            equity_tp2 = pd.DataFrame({"ts": trades_tp2["exit_ts"].values, "equity": eq})

    # Combined run (Gate ON + Gate OFF): merge trades and equity
    if run_flip and run_tp2:
        trades_flip = trades.copy()
        trades_flip["strategy"] = "on"
        trades_tp2_out = trades_tp2.copy()
        trades_tp2_out["strategy"] = "off"
        common = ["entry_ts", "exit_ts", "side", "entry_px", "exit_px", "exit_event", "pnl_pct", "strategy"]
        for c in common:
            if c not in trades_tp2_out.columns and c != "strategy":
                trades_tp2_out[c] = np.nan
        trades_combined = pd.concat(
            [trades_flip[common], trades_tp2_out[common]],
            ignore_index=True,
        ).sort_values("exit_ts").reset_index(drop=True)
        equity_combined = _equity_from_trades(trades_combined, initial_capital=1.0)
        trades = trades_combined
        equity = equity_combined
        print(f"INFO merged: flip trades={len(trades_flip)} tp2 trades={len(trades_tp2_out)} combined={len(trades)}")
    elif run_tp2 and not run_flip and len(trades_tp2):
        trades = trades_tp2.copy()
        trades["strategy"] = "off"
        equity = equity_tp2

    equity0 = 1.0
    equity1 = float(equity["equity"].iloc[-1]) if len(equity) else equity0
    total_return_pct = (equity1 / equity0 - 1.0) * 100.0
    if len(equity):
        peak = equity["equity"].cummax()
        dd = (equity["equity"] / peak - 1.0) * 100.0
        max_drawdown_pct = float(dd.min())
    else:
        max_drawdown_pct = 0.0

    entries_count = int((events["event"] == "entry").sum()) if events is not None and len(events) else 0
    flips_count = int(events["event"].isin(["tp_exit", "signal_flip_exit"]).sum()) if events is not None and len(events) else 0
    sl_count = int((events["event"] == "sl_exit").sum()) if events is not None and len(events) else 0

    # effective (engine clamps internally to <=50; mirror that here)
    effective_swing_lb = int(min(int(args.swing_lookback), 50))

    # =========================
    # REAL pricing via fills (optional; flip events only)
    # =========================
    fills_miss_rate = None
    trades_real = pd.DataFrame()
    equity_real = pd.DataFrame({"ts": [], "equity": []})
    total_return_pct_real = None
    max_drawdown_pct_real = None

    if args.fills_parquet and events is not None:
        fills = _read_fills_parquet(str(args.fills_parquet), str(args.fill_col))
        events2, fills_miss_rate = _map_events_to_fills_asof(events, fills, str(args.fill_col))

        trades_real = _pair_trades_from_events(events2, price_col="price_real")
        trades_real = _compute_trade_pnl_from_prices(
            trades_real,
            fee_bps=float(args.fee_bps),
            price_cols=("entry_px", "exit_px"),
        )

        trades_real_for_eq = trades_real.copy()
        trades_real_for_eq["pnl_pct"] = trades_real_for_eq["pnl_pct_real"]
        equity_real = _equity_from_trades(trades_real_for_eq, initial_capital=1.0)

        eq1r = float(equity_real["equity"].iloc[-1]) if len(equity_real) else equity0
        total_return_pct_real = (eq1r / equity0 - 1.0) * 100.0
        if len(equity_real):
            peak_r = equity_real["equity"].cummax()
            dd_r = (equity_real["equity"] / peak_r - 1.0) * 100.0
            max_drawdown_pct_real = float(dd_r.min())
        else:
            max_drawdown_pct_real = 0.0

        print(
            f"INFO fills mapped: col={args.fill_col} miss_rate={0.0 if fills_miss_rate is None else (100*fills_miss_rate):.4f}%"
        )

    stats = {
        "rows": int(len(bars)),
        "start": str(bars["ts"].min()) if len(bars) else None,
        "end": str(bars["ts"].max()) if len(bars) else None,
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "trades": int(len(trades)),
        "entries_count": entries_count,
        "flips_count": flips_count,
        "sl_exits_count": sl_count,
        "fee_bps_roundtrip": float(args.fee_bps),
        "min_sl_pct": float(args.min_sl_pct),
        "max_sl_pct": float(args.max_sl_pct),
        "swing_lookback_arg": int(args.swing_lookback),
        "swing_lookback_effective": effective_swing_lb,
        "ttp_trail_pct": float(args.ttp_trail_pct),
        "regime": ("external" if args.regime_csv else str(args.regime)),
        "regime_csv": (str(args.regime_csv) if args.regime_csv else None),
        "regime_csv_off": (str(args.regime_csv_off) if args.regime_csv_off else None),
        "run_flip": run_flip,
        "run_tp2": run_tp2,
        "signals_jsonl": str(args.signals_jsonl),
        "parquet": str(args.parquet),
        "box": float(args.box),
        "fills_parquet": (str(args.fills_parquet) if args.fills_parquet else None),
        "fill_col": (str(args.fill_col) if args.fills_parquet else None),
        "fills_miss_rate": (float(fills_miss_rate) if fills_miss_rate is not None else None),
        "trades_real": (int(len(trades_real)) if len(trades_real) else 0),
        "total_return_pct_real": (float(total_return_pct_real) if total_return_pct_real is not None else None),
        "max_drawdown_pct_real": (float(max_drawdown_pct_real) if max_drawdown_pct_real is not None else None),
    }
    if run_tp2 and events_tp2 is not None and hasattr(events_tp2, "attrs"):
        stats["tp2_entries"] = int(events_tp2.attrs.get("counts", {}).get("trades_entries", 0))
        stats["tp2_sl_exits"] = int(events_tp2.attrs.get("counts", {}).get("sl_exits", 0))
        stats["tp2_regime_exits"] = int(events_tp2.attrs.get("counts", {}).get("regime_exits", 0))

    print(f"INFO stats {stats}")
    print(f"INFO events {0 if events is None else len(events)}")
    if events is not None and len(events) > 0:
        print(f"INFO last event: {events.iloc[-1].to_dict()}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if events is not None:
        events.to_parquet(out_dir / "events.parquet", index=False)
    if events_tp2 is not None:
        events_tp2.to_parquet(out_dir / "events_tp2.parquet", index=False)
    if legs_tp2 is not None and len(legs_tp2):
        legs_tp2.to_parquet(out_dir / "legs.parquet", index=False)
    trades.to_parquet(out_dir / "trades.parquet", index=False)
    equity.to_parquet(out_dir / "equity.parquet", index=False)

    if args.fills_parquet and len(trades_real):
        trades_real.to_parquet(out_dir / "trades_real.parquet", index=False)
        equity_real.to_parquet(out_dir / "equity_real.parquet", index=False)

    if regime_on is not None:
        pd.DataFrame({"ts": regime_on.index.values, "regime_on": regime_on.astype(int).values}).to_parquet(
            out_dir / "regime.parquet", index=False
        )
    if gate_off_aligned is not None:
        pd.DataFrame({"ts": gate_off_aligned.index.values, "gate_off": gate_off_aligned.astype(int).values}).to_parquet(
            out_dir / "regime_off.parquet", index=False
        )

    print(f"INFO wrote {out_dir}")


if __name__ == "__main__":
    main()