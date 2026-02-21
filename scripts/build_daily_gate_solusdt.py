#!/usr/bin/env python3
# scripts/build_daily_gate_solusdt.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------
# Indicators on brick OHLC
# ----------------------------
def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def _wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()


def chop(df: pd.DataFrame, n: int = 14) -> pd.Series:
    n = int(n)
    tr = _true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = df["high"].astype(float).rolling(n, min_periods=n).max()
    ll = df["low"].astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    out = 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))
    return out


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
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
    out = _wilder_smooth(dx, n)
    return out


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Wilder ATR on brick OHLC.
    """
    n = int(n)
    tr = _true_range(df)
    return _wilder_smooth(tr, n)


# ----------------------------
# Hysteresis helpers (PER-ROW thresholds)
# ----------------------------
def hysteresis_high_is_good(x: pd.Series, on_th: pd.Series, off_th: pd.Series) -> pd.Series:
    """
    Per-step hysteresis with per-row thresholds.
    state turns ON when x[i] >= on_th[i]
    state turns OFF when x[i] <= off_th[i]
    """
    x = pd.to_numeric(x, errors="coerce")
    on_th = pd.to_numeric(on_th, errors="coerce")
    off_th = pd.to_numeric(off_th, errors="coerce")

    state = False
    out = []
    for v, onv, offv in zip(x.values, on_th.values, off_th.values):
        if np.isnan(v) or np.isnan(onv) or np.isnan(offv):
            out.append(state)
            continue
        if (not state) and (v >= onv):
            state = True
        elif state and (v <= offv):
            state = False
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")


def hysteresis_low_is_bad(x: pd.Series, on_th: pd.Series, off_th: pd.Series, start_on: bool = True) -> pd.Series:
    """
    Per-step hysteresis with per-row thresholds (low is good, high is bad).
    state turns OFF when x[i] >= off_th[i]
    state turns ON  when x[i] <= on_th[i]
    """
    x = pd.to_numeric(x, errors="coerce")
    on_th = pd.to_numeric(on_th, errors="coerce")
    off_th = pd.to_numeric(off_th, errors="coerce")

    state = bool(start_on)
    out = []
    for v, onv, offv in zip(x.values, on_th.values, off_th.values):
        if np.isnan(v) or np.isnan(onv) or np.isnan(offv):
            out.append(state)
            continue
        if state and (v >= offv):
            state = False
        elif (not state) and (v <= onv):
            state = True
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")


def grace_smooth_bool(x: pd.Series, grace_days: int = 3) -> pd.Series:
    """
    Keep gate ON for 'grace_days' after it would turn OFF (simple debouncer).
    """
    g = int(grace_days)
    if g <= 0:
        return x.astype(bool)

    out = []
    hold = 0
    for v in x.astype(bool).values:
        if v:
            hold = g
            out.append(True)
        else:
            if hold > 0:
                hold -= 1
                out.append(True)
            else:
                out.append(False)
    return pd.Series(out, index=x.index, dtype="bool")


# ----------------------------
# Walk-forward thresholding
# ----------------------------
@dataclass
class WFParams:
    train_days: int = 120
    step_days: int = 7
    dead_days: int = 3  # avoid lookahead: thresholds fit on data ending at (day - dead_days - 1)

    chon_q: float = 0.75
    choff_q: float = 0.65

    adx_q: float = 0.50
    adx_band: float = 0.05  # makes adx_on/adx_off via q±band

    use_atr: bool = True
    atr_q: float = 0.50
    atr_band: float = 0.05  # makes atr_on/atr_off via q±band

    grace_days: int = 3


def compute_daily_features(renko: pd.DataFrame, chop_len: int, adx_len: int, atr_len: int) -> pd.DataFrame:
    df = renko.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # indicators per brick
    df["CHOP"] = chop(df, n=chop_len)
    df["ADX"] = adx(df, n=adx_len)
    df["ATR"] = atr(df, n=atr_len)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    df["ATR_PCT"] = (df["ATR"].astype(float) / close.replace(0.0, np.nan)).astype(float)

    # daily aggregate (mean)
    df["day"] = df["ts"].dt.floor("D")
    daily = df.groupby("day", as_index=False).agg(
        CHOP_mean=("CHOP", "mean"),
        ADX_mean=("ADX", "mean"),
        ATRPCT_mean=("ATR_PCT", "mean"),
        bricks=("ts", "count"),
    )
    daily = daily.dropna(subset=["CHOP_mean", "ADX_mean", "ATRPCT_mean"]).reset_index(drop=True)
    daily = daily.rename(columns={"day": "ts"})
    return daily


def _qpair(center_q: float, band: float) -> tuple[float, float]:
    q_lo = max(0.0, min(1.0, float(center_q) - float(band)))
    q_hi = max(0.0, min(1.0, float(center_q) + float(band)))
    return q_lo, q_hi


def wf_daily_gate(daily: pd.DataFrame, p: WFParams) -> pd.DataFrame:
    d = daily.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d.sort_values("ts").reset_index(drop=True)

    out_rows = []
    last_fit_at: Optional[pd.Timestamp] = None

    cur = {
        "chop_on": np.nan,
        "chop_off": np.nan,
        "adx_on": np.nan,
        "adx_off": np.nan,
        "atr_on": np.nan,
        "atr_off": np.nan,
    }

    for i in range(len(d)):
        day = pd.Timestamp(d.loc[i, "ts"])

        # refit thresholds every step_days
        need_refit = last_fit_at is None or (day - last_fit_at).days >= p.step_days

        if need_refit:
            fit_end_idx = i - p.dead_days - 1
            fit_start_idx = fit_end_idx - p.train_days + 1

            if fit_start_idx >= 0 and fit_end_idx >= 0:
                hist = d.iloc[fit_start_idx : fit_end_idx + 1].copy()

                ch = hist["CHOP_mean"].astype(float)
                ax = hist["ADX_mean"].astype(float)
                at = hist["ATRPCT_mean"].astype(float)

                cur["chop_on"] = float(ch.quantile(p.chon_q))
                cur["chop_off"] = float(ch.quantile(p.choff_q))

                q_lo, q_hi = _qpair(p.adx_q, p.adx_band)
                cur["adx_on"] = float(ax.quantile(q_lo))
                cur["adx_off"] = float(ax.quantile(q_hi))

                if p.use_atr:
                    q_lo2, q_hi2 = _qpair(p.atr_q, p.atr_band)
                    cur["atr_on"] = float(at.quantile(q_lo2))
                    cur["atr_off"] = float(at.quantile(q_hi2))

                last_fit_at = day

        out_rows.append(
            {
                "ts": day,
                "CHOP_mean": float(d.loc[i, "CHOP_mean"]),
                "ADX_mean": float(d.loc[i, "ADX_mean"]),
                "ATRPCT_mean": float(d.loc[i, "ATRPCT_mean"]),
                "bricks": int(d.loc[i, "bricks"]),
                "chop_on": cur["chop_on"],
                "chop_off": cur["chop_off"],
                "adx_on": cur["adx_on"],
                "adx_off": cur["adx_off"],
                "atr_on": cur["atr_on"],
                "atr_off": cur["atr_off"],
                "fit_ts": (last_fit_at if last_fit_at is not None else pd.NaT),
            }
        )

    o = pd.DataFrame(out_rows)

    # require fitted thresholds
    req = ["chop_on", "chop_off", "adx_on", "adx_off"]
    if p.use_atr:
        req += ["atr_on", "atr_off"]
    o = o.dropna(subset=req).reset_index(drop=True)

    # CHOP: high is good (range-y)
    chop_ok = hysteresis_high_is_good(o["CHOP_mean"], on_th=o["chop_on"], off_th=o["chop_off"])

    # ADX: low is good
    adx_ok = hysteresis_low_is_bad(o["ADX_mean"], on_th=o["adx_on"], off_th=o["adx_off"], start_on=True)

    gate = chop_ok & adx_ok

    # ATR_PCT: low is good (avoid high vol regimes for flip-strats)
    if p.use_atr:
        atr_ok = hysteresis_low_is_bad(o["ATRPCT_mean"], on_th=o["atr_on"], off_th=o["atr_off"], start_on=True)
        gate = gate & atr_ok

    gate = grace_smooth_bool(gate.astype(bool), grace_days=p.grace_days)

    o["gate_on"] = gate.astype(int)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renko-parquet", required=True, help="Renko OHLC parquet with ts,open,high,low,close")
    ap.add_argument("--out-csv", required=True, help="Output daily gate CSV path")

    ap.add_argument("--chop-len", type=int, default=14)
    ap.add_argument("--adx-len", type=int, default=14)
    ap.add_argument("--atr-len", type=int, default=14)

    ap.add_argument("--train-days", type=int, default=120)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--dead-days", type=int, default=3)

    ap.add_argument("--chon-q", type=float, default=0.75)
    ap.add_argument("--choff-q", type=float, default=0.65)

    ap.add_argument("--adx-q", type=float, default=0.50)
    ap.add_argument("--adx-band", type=float, default=0.05)

    ap.add_argument("--use-atr", action="store_true", help="Enable ATR_PCT gate (recommended)")
    ap.add_argument("--atr-q", type=float, default=0.50)
    ap.add_argument("--atr-band", type=float, default=0.05)

    ap.add_argument("--grace-days", type=int, default=3)
    args = ap.parse_args()

    renko = pd.read_parquet(args.renko_parquet)
    if "ts" not in renko.columns:
        if isinstance(renko.index, pd.DatetimeIndex):
            renko = renko.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("renko parquet missing ts column and not datetime indexed")

    need = {"open", "high", "low", "close"}
    missing = need - set(renko.columns)
    if missing:
        raise ValueError(f"renko parquet missing columns: {sorted(missing)}")

    daily = compute_daily_features(
        renko,
        chop_len=int(args.chop_len),
        adx_len=int(args.adx_len),
        atr_len=int(args.atr_len),
    )

    p = WFParams(
        train_days=int(args.train_days),
        step_days=int(args.step_days),
        dead_days=int(args.dead_days),
        chon_q=float(args.chon_q),
        choff_q=float(args.choff_q),
        adx_q=float(args.adx_q),
        adx_band=float(args.adx_band),
        use_atr=bool(args.use_atr),
        atr_q=float(args.atr_q),
        atr_band=float(args.atr_band),
        grace_days=int(args.grace_days),
    )

    out = wf_daily_gate(daily, p=p)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    on_rate = 100.0 * float(out["gate_on"].mean()) if len(out) else 0.0
    print(f"INFO wrote {out_path} rows={len(out)} ON-rate={on_rate:.2f}%")
    print("INFO columns:", list(out.columns))


if __name__ == "__main__":
    main()
