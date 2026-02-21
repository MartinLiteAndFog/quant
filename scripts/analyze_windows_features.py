# scripts/analyze_windows_features.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------- Indicators ----------

def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def _choppiness(df: pd.DataFrame, n: int) -> pd.Series:
    """
    CHOP = 100 * log10( sum(TR,n) / (HH(n)-LL(n)) ) / log10(n)
    """
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


def _efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """
    ER = abs(close - close.shift(n)) / sum(abs(diff(close)), n)
    """
    n = int(n)
    net = (close - close.shift(n)).abs()
    denom = close.diff().abs().rolling(n, min_periods=n).sum().replace(0.0, np.nan)
    return net / denom


def _atr_pct(df: pd.DataFrame, n: int) -> pd.Series:
    """
    ATR% = ATR(n) / close * 100
    """
    n = int(n)
    tr = _true_range(df)
    atr = _wilder_smooth(tr, n)
    return (atr / df["close"].astype(float)) * 100.0


def _rvol_pct(df: pd.DataFrame, n: int) -> pd.Series:
    """
    RVOL%: rolling std of 1m returns (n) * 100
    """
    n = int(n)
    ret = df["close"].astype(float).pct_change()
    return ret.rolling(n, min_periods=n).std() * 100.0


def _rsi(close: pd.Series, n: int) -> pd.Series:
    n = int(n)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_smooth(gain, n)
    avg_loss = _wilder_smooth(loss, n)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _trend_slope_and_r2(close: pd.Series, n: int) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling OLS on log(close) vs time index [0..n-1]:
      slope: per-bar slope of log price
      r2: fit quality
    """
    n = int(n)
    y = np.log(close.astype(float).replace(0.0, np.nan))
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    slopes = np.full(len(y), np.nan, dtype=float)
    r2s = np.full(len(y), np.nan, dtype=float)

    yv = y.values
    for i in range(n - 1, len(yv)):
        w = yv[i - n + 1 : i + 1]
        if np.any(~np.isfinite(w)):
            continue
        y_mean = w.mean()
        cov = ((x - x_mean) * (w - y_mean)).sum()
        slope = cov / x_var
        # r2
        y_hat = y_mean + slope * (x - x_mean)
        ss_res = ((w - y_hat) ** 2).sum()
        ss_tot = ((w - y_mean) ** 2).sum()
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        slopes[i] = slope
        r2s[i] = r2

    return pd.Series(slopes, index=close.index), pd.Series(r2s, index=close.index)


# ---------- IO / Summaries ----------

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


def _q(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0}
    qs = np.quantile(s.values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(s)),
        "mean": float(np.mean(s.values)),
        "p05": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
    }


@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_windows() -> List[Window]:
    # Your marked positive phases (inclusive dates)
    raw = [
        ("2024-01-24", "2024-02-13"),
        ("2024-02-27", "2024-03-15"),
        ("2024-03-26", "2024-04-12"),
        ("2024-05-28", "2024-07-03"),
        ("2024-10-25", "2024-12-10"),
        ("2024-12-24", "2025-01-02"),
        ("2025-09-16", "2025-10-23"),
        ("2025-11-07", "2025-12-12"),
    ]
    out = []
    for s, e in raw:
        start = pd.Timestamp(f"{s}T00:00:00Z")
        end = pd.Timestamp(f"{e}T23:59:59Z")  # inclusive
        out.append(Window(start=start, end=end))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)

    ap.add_argument("--out", required=True)

    ap.add_argument("--chop-len", type=int, default=14)
    ap.add_argument("--adx-len", type=int, default=14)
    ap.add_argument("--er-len", type=int, default=120)
    ap.add_argument("--atr-len", type=int, default=14)
    ap.add_argument("--rvol-len", type=int, default=20)
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--trend-len", type=int, default=240)

    args = ap.parse_args()

    bars = _read_ohlcv_parquet(args.parquet)

    # IMPORTANT: make sure comparisons are tz-aware (fixes your TypeError)
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars = bars.sort_values("ts").reset_index(drop=True)

    # compute full-series features once
    feat = pd.DataFrame({"ts": bars["ts"].copy()})
    feat["CHOP"] = _choppiness(bars, args.chop_len)
    feat["ADX"] = _adx(bars, args.adx_len)
    feat["ER"] = _efficiency_ratio(bars["close"].astype(float), args.er_len)
    feat["ATR_PCT"] = _atr_pct(bars, args.atr_len)
    feat["RVOL_PCT"] = _rvol_pct(bars, args.rvol_len)
    feat["RSI"] = _rsi(bars["close"].astype(float), args.rsi_len)
    slope, r2 = _trend_slope_and_r2(bars["close"].astype(float), args.trend_len)
    feat["TREND_SLOPE"] = slope
    feat["TREND_R2"] = r2

    windows = _parse_windows()
    rows = []
    for i, w in enumerate(windows, start=1):
        m = (feat["ts"] >= w.start) & (feat["ts"] <= w.end)
        sub = feat.loc[m].copy()

        row = {
            "window_id": i,
            "start": w.start.isoformat(),
            "end": w.end.isoformat(),
            "rows": int(len(sub)),
        }

        for col in ["CHOP", "ADX", "ER", "ATR_PCT", "RVOL_PCT", "RSI", "TREND_SLOPE", "TREND_R2"]:
            q = _q(sub[col])
            for k, v in q.items():
                row[f"{col}_{k}"] = v

        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("WROTE", out_path)
    print(out[["window_id", "start", "end", "rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
