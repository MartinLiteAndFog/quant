# regime_window.py
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers / Indicators
# -----------------------------
def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = _ema(up, length) / _ema(dn, length).replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _ema(tr, length)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    # Wilder-ish via EMA to keep it simple/stable.
    up_move = high.diff()
    dn_move = -low.diff()

    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_ = _ema(tr, length)
    plus_di = 100.0 * (_ema(pd.Series(plus_dm, index=high.index), length) / atr_.replace(0, np.nan))
    minus_di = 100.0 * (_ema(pd.Series(minus_dm, index=high.index), length) / atr_.replace(0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return _ema(dx, length)


def chop(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    # Choppiness Index (CHOP)
    # CHOP = 100 * log10( sum(ATR(1), n) / (max(high,n) - min(low,n)) ) / log10(n)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    sum_tr = tr.rolling(length).sum()
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    denom = (hh - ll).replace(0, np.nan)

    val = 100.0 * (np.log10(sum_tr / denom) / np.log10(length))
    return val


def efficiency_ratio(close: pd.Series, length: int = 120) -> pd.Series:
    # ER = |close - close[n]| / sum(|diff|, n)
    change = (close - close.shift(length)).abs()
    vol = close.diff().abs().rolling(length).sum().replace(0, np.nan)
    return change / vol


def rolling_rvol_pct(volume: pd.Series, length: int = 120) -> pd.Series:
    # Relative volume deviation vs rolling mean: (vol/mean - 1)
    m = volume.rolling(length).mean().replace(0, np.nan)
    return (volume / m) - 1.0


def rolling_trend(close: pd.Series, length: int = 120) -> Tuple[pd.Series, pd.Series]:
    # Linear regression slope + R^2 on rolling window (time index 0..n-1)
    x = np.arange(length, dtype=float)

    def _fit(y: np.ndarray) -> Tuple[float, float]:
        if np.any(~np.isfinite(y)):
            return (np.nan, np.nan)
        y_mean = y.mean()
        x_mean = x.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var = ((x - x_mean) ** 2).sum()
        if var == 0:
            return (np.nan, np.nan)
        slope = cov / var
        intercept = y_mean - slope * x_mean
        y_hat = intercept + slope * x
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        return (slope, r2)

    slopes = np.full(len(close), np.nan, dtype=float)
    r2s = np.full(len(close), np.nan, dtype=float)
    y = close.to_numpy(dtype=float)

    for i in range(length - 1, len(y)):
        s, r2 = _fit(y[i - length + 1 : i + 1])
        slopes[i] = s
        r2s[i] = r2

    return pd.Series(slopes, index=close.index), pd.Series(r2s, index=close.index)


# -----------------------------
# IO: read OHLCV parquet
# -----------------------------
def _read_parquet_any(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    # normalize timestamp
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
    elif df.index.name in ("ts", "timestamp", "time"):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        # try common
        for c in ("timestamp", "time", "date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True)
                df = df.set_index(c)
                break
    df = df.sort_index()
    return df


def load_ohlcv_from_dir(raw_dir: Path, start: str, end: str) -> pd.DataFrame:
    # Accept either:
    # - a single parquet file
    # - a directory containing many parquet files
    if raw_dir.is_file():
        df = _read_parquet_any(raw_dir)
        return df.loc[pd.to_datetime(start, utc=True) : pd.to_datetime(end, utc=True)]

    files = sorted(raw_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {raw_dir}")

    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    parts = []
    for p in files:
        try:
            d = _read_parquet_any(p)
        except Exception:
            continue
        if len(d) == 0:
            continue
        # quick overlap check
        if d.index.max() < start_ts or d.index.min() > end_ts:
            continue
        parts.append(d)

    if not parts:
        raise FileNotFoundError(f"Found parquet files, but none overlap [{start_ts} .. {end_ts}] in {raw_dir}")

    df = pd.concat(parts).sort_index()
    df = df.loc[start_ts:end_ts]
    df = df[~df.index.duplicated(keep="last")]
    return df


# -----------------------------
# Summary
# -----------------------------
@dataclass
class Summary:
    n: int
    start: str
    end: str
    means: dict
    pcts: dict


def summarize(series: pd.Series) -> Tuple[dict, dict]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {}, {}
    means = {"mean": float(s.mean()), "std": float(s.std(ddof=0))}
    pcts = {f"p{q}": float(s.quantile(q / 100.0)) for q in (5, 10, 25, 50, 75, 90, 95)}
    return means, pcts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Parquet file OR directory (raw OHLCV) to scan")
    ap.add_argument("--start", required=True, help="Start UTC, e.g. 2025-02-01")
    ap.add_argument("--end", required=True, help="End UTC, e.g. 2025-04-30")
    ap.add_argument("--out", default="regime_summary_2025_02_04.json", help="Output json path")
    ap.add_argument("--chop_n", type=int, default=14)
    ap.add_argument("--adx_n", type=int, default=14)
    ap.add_argument("--er_n", type=int, default=120)
    ap.add_argument("--rsi_n", type=int, default=14)
    ap.add_argument("--trend_n", type=int, default=120)
    ap.add_argument("--rvol_n", type=int, default=120)
    args = ap.parse_args()

    df = load_ohlcv_from_dir(Path(args.raw), args.start, args.end)

    need = {"open", "high", "low", "close"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns {missing} in data from {args.raw}. Columns: {list(df.columns)[:30]}")

    # volume is optional
    if "volume" not in df.columns:
        df["volume"] = np.nan

    hi, lo, cl = df["high"], df["low"], df["close"]
    vol = df["volume"]

    s_chop = chop(hi, lo, cl, args.chop_n)
    s_adx = adx(hi, lo, cl, args.adx_n)
    s_er = efficiency_ratio(cl, args.er_n)
    s_atr = atr(hi, lo, cl, 14)
    s_atr_pct = (s_atr / cl.replace(0, np.nan))  # fraction (not *100), matches your earlier style
    s_rsi = rsi(cl, args.rsi_n)
    s_rvol_pct = rolling_rvol_pct(vol, args.rvol_n) if vol.notna().any() else pd.Series(np.nan, index=df.index)
    s_slope, s_r2 = rolling_trend(cl, args.trend_n)

    payload = {
        "window": {"start": str(df.index.min()), "end": str(df.index.max()), "rows": int(len(df))},
        "means": {},
        "pcts": {},
    }

    for name, ser in [
        ("CHOP", s_chop),
        ("ADX", s_adx),
        ("ER", s_er),
        ("ATR_PCT", s_atr_pct),
        ("RVOL_PCT", s_rvol_pct),
        ("RSI", s_rsi),
        ("TREND_SLOPE", s_slope),
        ("TREND_R2", s_r2),
    ]:
        m, p = summarize(ser)
        payload["means"][name] = m
        payload["pcts"][name] = p

    out = Path(args.out)
    out.write_text(pd.Series(payload).to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    print(out.read_text(encoding='utf-8'))


if __name__ == "__main__":
    main()
