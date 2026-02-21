# scripts/scan_regime_indicators.py
"""
Scan a target phase (e.g. full January) across multiple "weather/regime" indicators,
then search the rest of the dataset for windows with similar indicator fingerprints.

Goal:
- Compute indicators (CHOP, ADX, ER, ATR%, RVOL, RSI, TrendSlope, TrendR2)
- Summarize a target window (mean + p25/p50/p75) for each indicator
- Slide a window over a scan range and compute the same summaries
- Rank candidate windows by distance to the target fingerprint
- Write top matches to CSV (+ save the target fingerprint JSON)

Example:
python scripts/scan_regime_indicators.py \
  --parquet /Users/martinpeter/quant/data/clean/SOL-USDC_1m_20260207T143630Z.clean4.parquet \
  --target-start 2026-01-01 \
  --target-end   2026-02-01 \
  --window-days 31 \
  --step-hours 6 \
  --top-k 50 \
  --out data/regime_scan/jan2026_matches.csv

Notes:
- Works best if the parquet is already 1m bars.
- If your parquet has multiple years (as yours does), this is intended.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# IO
# ----------------------------

def read_ohlcv_parquet(path: str) -> pd.DataFrame:
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


# ----------------------------
# Indicators
# ----------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr


def wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    # EMA alpha=1/n (Wilder)
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    n = int(n)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up = high.diff()
    down = -low.diff()

    dm_plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = true_range(df)
    atr = wilder_smooth(tr, n)
    sm_plus = wilder_smooth(dm_plus, n)
    sm_minus = wilder_smooth(dm_minus, n)

    di_plus = 100.0 * (sm_plus / atr.replace(0.0, np.nan))
    di_minus = 100.0 * (sm_minus / atr.replace(0.0, np.nan))

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan)
    return wilder_smooth(dx, n)


def choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    CHOP = 100 * log10( sum(TR,n) / (HH(n)-LL(n)) ) / log10(n)
    Higher => choppier/rangier
    """
    n = int(n)
    tr = true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = df["high"].astype(float).rolling(n, min_periods=n).max()
    ll = df["low"].astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    chop = 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))
    return chop


def efficiency_ratio(close: pd.Series, n: int = 40) -> pd.Series:
    """
    Kaufman Efficiency Ratio (ER):
      ER = |close - close_n| / sum(|diff(close)| over n)
    Range ~ [0..1] : 0 choppy, 1 trend
    """
    n = int(n)
    c = close.astype(float)
    net = (c - c.shift(n)).abs()
    denom = c.diff().abs().rolling(n, min_periods=n).sum()
    return net / denom.replace(0.0, np.nan)


def atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    atr = wilder_smooth(true_range(df), int(n))
    close = df["close"].astype(float)
    return (atr / close.replace(0.0, np.nan)) * 100.0


def realized_vol_pct(close: pd.Series, n: int = 60) -> pd.Series:
    """
    Realized vol from log returns, in % per sqrt(bar).
    (Not annualized; we just want a regime proxy.)
    """
    n = int(n)
    c = close.astype(float)
    r = np.log(c / c.shift(1))
    rv = r.rolling(n, min_periods=n).std()
    return rv * 100.0


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    n = int(n)
    c = close.astype(float)
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = wilder_smooth(up, n)
    ad = wilder_smooth(dn, n)
    rs = au / ad.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def trend_slope_r2(close: pd.Series, n: int = 240) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling linear regression on log(close) vs time index 0..n-1.
    Returns:
      slope_per_bar (approx), r2
    """
    n = int(n)
    c = close.astype(float)
    y = np.log(c.replace(0.0, np.nan))

    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_d = x - x_mean
    sxx = (x_d ** 2).sum()

    # Rolling sums needed:
    # slope = cov(x,y)/var(x) where cov uses centered x
    # cov(x,y) = sum((x-xm)*(y-ym)) = sum((x-xm)*y) because sum(x-xm)=0
    # and ym is rolling mean(y)
    y_mean = y.rolling(n, min_periods=n).mean()
    sum_xc_y = y.rolling(n, min_periods=n).apply(
        lambda arr: float(np.dot(x_d, arr)),
        raw=True
    )
    slope = sum_xc_y / sxx

    # r2: 1 - SSE/SST
    # SST = sum((y-ym)^2)
    sst = y.rolling(n, min_periods=n).var() * (n - 1)
    # SSE via: SSE = sum((y - (a + b*x))^2). Compute a=ym - b*xm
    # We'll compute predicted and SSE with another rolling apply (still OK for n~240 with step-hours scanning).
    def _sse(arr: np.ndarray) -> float:
        arr = arr.astype(float)
        if np.any(~np.isfinite(arr)):
            return np.nan
        ym = arr.mean()
        b = float(np.dot(x_d, arr)) / sxx
        a = ym - b * x_mean
        yhat = a + b * x
        e = arr - yhat
        return float(np.dot(e, e))

    sse = y.rolling(n, min_periods=n).apply(_sse, raw=True)
    r2 = 1.0 - (sse / sst.replace(0.0, np.nan))
    return slope, r2


# ----------------------------
# Window fingerprinting
# ----------------------------

@dataclass
class FingerprintSpec:
    stats: Tuple[str, ...] = ("mean", "p25", "p50", "p75")

def window_stats(df: pd.DataFrame, cols: List[str], spec: FingerprintSpec) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            for st in spec.stats:
                out[f"{c}:{st}"] = float("nan")
            continue
        out[f"{c}:mean"] = float(s.mean())
        out[f"{c}:p25"] = float(s.quantile(0.25))
        out[f"{c}:p50"] = float(s.quantile(0.50))
        out[f"{c}:p75"] = float(s.quantile(0.75))
    return out


def fingerprint_distance(a: Dict[str, float], b: Dict[str, float], scales: Dict[str, float]) -> float:
    # Robust normalized L2 using per-feature scale (e.g. IQR of target)
    acc = 0.0
    n = 0
    for k, av in a.items():
        bv = b.get(k, np.nan)
        if not (np.isfinite(av) and np.isfinite(bv)):
            continue
        sc = scales.get(k, 1.0)
        sc = sc if (np.isfinite(sc) and sc > 1e-12) else 1.0
        z = (av - bv) / sc
        acc += float(z * z)
        n += 1
    if n == 0:
        return float("inf")
    return float(math.sqrt(acc / n))


def build_scales_from_target(target_fp: Dict[str, float], target_window: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    # Use IQR per indicator as scale for ALL stats of that indicator
    scales: Dict[str, float] = {}
    for c in cols:
        s = pd.to_numeric(target_window[c], errors="coerce").dropna()
        if len(s) == 0:
            iqr = 1.0
        else:
            iqr = float(s.quantile(0.75) - s.quantile(0.25))
            if not (np.isfinite(iqr) and iqr > 1e-12):
                iqr = float(s.std()) if np.isfinite(float(s.std())) and float(s.std()) > 1e-12 else 1.0
        for st in ("mean", "p25", "p50", "p75"):
            scales[f"{c}:{st}"] = iqr
    return scales


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)

    # Target phase (e.g. full January)
    ap.add_argument("--target-start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--target-end", required=True, help="YYYY-MM-DD (UTC), end-exclusive")

    # Scan range (defaults to whole dataset)
    ap.add_argument("--scan-start", default=None, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--scan-end", default=None, help="YYYY-MM-DD (UTC), end-exclusive")

    # Windowing
    ap.add_argument("--window-days", type=int, default=31, help="length of candidate window")
    ap.add_argument("--step-hours", type=int, default=6, help="advance step for candidate endpoints")

    # Indicator lengths
    ap.add_argument("--chop-len", type=int, default=14)
    ap.add_argument("--adx-len", type=int, default=14)
    ap.add_argument("--er-len", type=int, default=40)
    ap.add_argument("--atr-len", type=int, default=14)
    ap.add_argument("--rvol-len", type=int, default=60)
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--trend-len", type=int, default=240)

    # Output
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", required=True, help="CSV path for matches")
    ap.add_argument("--out-target-json", default=None, help="optional JSON path to write target fingerprint")
    ap.add_argument("--exclude-target", action="store_true", help="exclude windows overlapping the target period")

    args = ap.parse_args()

    df = read_ohlcv_parquet(args.parquet)
    df = df[["ts", "open", "high", "low", "close"]].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Filter scan range early (but keep enough history for indicators)
    ts0 = df["ts"].min()
    ts1 = df["ts"].max()

    target_start = pd.Timestamp(args.target_start, tz="UTC")
    target_end = pd.Timestamp(args.target_end, tz="UTC")

    scan_start = pd.Timestamp(args.scan_start, tz="UTC") if args.scan_start else ts0
    scan_end = pd.Timestamp(args.scan_end, tz="UTC") if args.scan_end else ts1 + pd.Timedelta(minutes=1)

    # Compute indicators on the FULL filtered df (scan range expanded a bit for rolling warmup)
    # Warmup: max lookback
    warm_n = max(args.chop_len, args.adx_len, args.er_len, args.atr_len, args.rvol_len, args.rsi_len, args.trend_len) + 5
    warm_td = pd.Timedelta(minutes=warm_n * 2)  # conservative
    compute_start = min(scan_start, target_start) - pd.Timedelta(days=args.window_days) - warm_td
    compute_end = max(scan_end, target_end) + pd.Timedelta(minutes=1)

    dfx = df[(df["ts"] >= compute_start) & (df["ts"] < compute_end)].copy().reset_index(drop=True)
    close = dfx["close"].astype(float)

    dfx["CHOP"] = choppiness(dfx, n=args.chop_len)
    dfx["ADX"] = adx(dfx, n=args.adx_len)
    dfx["ER"] = efficiency_ratio(close, n=args.er_len)
    dfx["ATR_PCT"] = atr_pct(dfx, n=args.atr_len)
    dfx["RVOL_PCT"] = realized_vol_pct(close, n=args.rvol_len)
    dfx["RSI"] = rsi(close, n=args.rsi_len)
    slope, r2 = trend_slope_r2(close, n=args.trend_len)
    dfx["TREND_SLOPE"] = slope
    dfx["TREND_R2"] = r2

    ind_cols = ["CHOP", "ADX", "ER", "ATR_PCT", "RVOL_PCT", "RSI", "TREND_SLOPE", "TREND_R2"]

    # Extract target window from indicator df
    tgt = dfx[(dfx["ts"] >= target_start) & (dfx["ts"] < target_end)].copy()
    if len(tgt) == 0:
        raise SystemExit("Target window is empty (check target-start/target-end vs parquet range).")

    spec = FingerprintSpec()
    target_fp = window_stats(tgt, ind_cols, spec)
    scales = build_scales_from_target(target_fp, tgt, ind_cols)

    # Candidate endpoints
    window_td = pd.Timedelta(days=int(args.window_days))
    step_td = pd.Timedelta(hours=int(args.step_hours))

    # Work on scan subset for endpoints
    scan_df = dfx[(dfx["ts"] >= scan_start) & (dfx["ts"] < scan_end)].copy()
    if len(scan_df) == 0:
        raise SystemExit("Scan range is empty (check scan-start/scan-end).")

    # Build a list of candidate end timestamps sampled every step_hours
    # Use floor to step grid starting at scan_start + window_td so we have full window.
    first_end = scan_start + window_td
    if first_end < scan_df["ts"].min():
        first_end = scan_df["ts"].min()

    ends: List[pd.Timestamp] = []
    t = first_end
    last_end = scan_end
    while t < last_end:
        ends.append(t)
        t += step_td

    # Evaluate candidates
    rows = []
    for end_ts in ends:
        start_ts = end_ts - window_td
        w = dfx[(dfx["ts"] >= start_ts) & (dfx["ts"] < end_ts)]
        if len(w) < 0.85 * len(tgt):  # rough guard: similar sample size
            continue

        # Optionally exclude windows overlapping target
        if args.exclude_target:
            if not (end_ts <= target_start or start_ts >= target_end):
                continue

        fp = window_stats(w, ind_cols, spec)
        dist = fingerprint_distance(fp, target_fp, scales)

        row = {
            "start": str(start_ts),
            "end": str(end_ts),
            "distance": float(dist),
            "rows": int(len(w)),
        }
        # also store mean values for quick reading
        for c in ind_cols:
            row[f"{c}_mean"] = fp.get(f"{c}:mean", np.nan)
            row[f"{c}_p50"] = fp.get(f"{c}:p50", np.nan)
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("distance", ascending=True).head(int(args.top_k)).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if args.out_target_json:
        tj = Path(args.out_target_json)
        tj.parent.mkdir(parents=True, exist_ok=True)
        tj.write_text(json.dumps({"target_fp": target_fp, "scales": scales}, indent=2), encoding="utf-8")

    print(f"OK target {target_start}..{target_end} rows={len(tgt)}")
    print(f"OK scan   {scan_start}..{scan_end} candidates={len(rows)} step_hours={args.step_hours} window_days={args.window_days}")
    print(f"WROTE {out_path}")
    if args.out_target_json:
        print(f"WROTE {args.out_target_json}")
    if len(out_df):
        best = out_df.iloc[0].to_dict()
        print(f"BEST distance={best['distance']:.4f} window={best['start']} .. {best['end']}")


if __name__ == "__main__":
    main()
