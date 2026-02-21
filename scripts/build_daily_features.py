# scripts/build_daily_features.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = _ema(up, length) / _ema(dn, length).replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return _ema(tr, length)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up_move = high.diff()
    dn_move = -low.diff()

    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)

    atr_ = _ema(tr, length)
    plus_di = 100.0 * (_ema(pd.Series(plus_dm, index=high.index), length) / atr_.replace(0, np.nan))
    minus_di = 100.0 * (_ema(pd.Series(minus_dm, index=high.index), length) / atr_.replace(0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return _ema(dx, length)


def chop(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    sum_tr = tr.rolling(length).sum()
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    denom = (hh - ll).replace(0, np.nan)
    return 100.0 * (np.log10(sum_tr / denom) / np.log10(length))


def efficiency_ratio(close: pd.Series, length: int = 120) -> pd.Series:
    change = (close - close.shift(length)).abs()
    vol = close.diff().abs().rolling(length).sum().replace(0, np.nan)
    return change / vol


def rolling_rvol_pct(volume: pd.Series, length: int = 120) -> pd.Series:
    m = volume.rolling(length).mean().replace(0, np.nan)
    return (volume / m) - 1.0


def rolling_trend(close: pd.Series, length: int = 120) -> Tuple[pd.Series, pd.Series]:
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


def read_parquet_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Renko OHLC parquet (ts, open, high, low, close[, volume])")
    ap.add_argument("--out", required=True, help="Output parquet (daily features)")
    ap.add_argument("--chop_n", type=int, default=14)
    ap.add_argument("--adx_n", type=int, default=14)
    ap.add_argument("--er_n", type=int, default=120)
    ap.add_argument("--rsi_n", type=int, default=14)
    ap.add_argument("--trend_n", type=int, default=120)
    ap.add_argument("--rvol_n", type=int, default=120)
    args = ap.parse_args()

    df = read_parquet_ohlc(Path(args.parquet))
    need = {"open", "high", "low", "close"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns {missing}. cols={list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    hi, lo, cl, vol = df["high"], df["low"], df["close"], df["volume"]

    feat = pd.DataFrame(index=df.index)
    feat["CHOP"] = chop(hi, lo, cl, args.chop_n)
    feat["ADX"] = adx(hi, lo, cl, args.adx_n)
    feat["ER"] = efficiency_ratio(cl, args.er_n)
    _atr = atr(hi, lo, cl, 14)
    feat["ATR_PCT"] = _atr / cl.replace(0, np.nan)
    feat["RSI"] = rsi(cl, args.rsi_n)
    feat["RVOL_PCT"] = rolling_rvol_pct(vol, args.rvol_n) if vol.notna().any() else np.nan
    slope, r2 = rolling_trend(cl, args.trend_n)
    feat["TREND_SLOPE"] = slope
    feat["TREND_R2"] = r2

    # Resample to daily: last known value each day (00:00..23:59), label at midnight UTC
    daily = feat.resample("1D").last().dropna(how="all")
    daily.index.name = "ts"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out)
    print("wrote", out, "rows", len(daily), "cols", list(daily.columns))


if __name__ == "__main__":
    main()

