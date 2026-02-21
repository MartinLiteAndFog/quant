# scripts/mark_and_analyze_windows.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Indicators / features
# -----------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def _wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()


def _choppiness(df: pd.DataFrame, n: int) -> pd.Series:
    n = int(n)
    tr = _true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = df["high"].astype(float).rolling(n, min_periods=n).max()
    ll = df["low"].astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    chop = 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))
    return chop


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


def _er_kaufman(close: pd.Series, n: int) -> pd.Series:
    """
    Kaufman Efficiency Ratio:
      ER = |close - close.shift(n)| / sum(|diff(close)|, n)
    """
    n = int(n)
    change = (close - close.shift(n)).abs()
    vol = close.diff().abs().rolling(n, min_periods=n).sum()
    return change / vol.replace(0.0, np.nan)


def _atr_pct(df: pd.DataFrame, n: int) -> pd.Series:
    tr = _true_range(df)
    atr = _wilder_smooth(tr, int(n))
    close = df["close"].astype(float)
    return (atr / close.replace(0.0, np.nan)).astype(float)


def _rvol_pct(df: pd.DataFrame, n: int) -> pd.Series:
    """
    Relative volume vs rolling median, expressed as pct above median:
      RVOL_PCT = (vol / median(vol,n)) - 1
    """
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    vol = pd.to_numeric(df["volume"], errors="coerce")
    med = vol.rolling(int(n), min_periods=int(n)).median()
    return (vol / med.replace(0.0, np.nan)) - 1.0


def _rsi(close: pd.Series, n: int) -> pd.Series:
    n = int(n)
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = _wilder_smooth(up, n) / _wilder_smooth(dn, n).replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _trend_slope_and_r2(close: pd.Series, n: int) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling OLS slope (per bar) and R^2 on log-price to make it scale-stable.
    """
    n = int(n)
    y = np.log(close.replace(0.0, np.nan).astype(float))
    idx = np.arange(n, dtype=float)

    slopes = np.full(len(y), np.nan, dtype=float)
    r2s = np.full(len(y), np.nan, dtype=float)

    for i in range(n - 1, len(y)):
        w = y.iloc[i - n + 1 : i + 1].values
        if np.any(np.isnan(w)):
            continue
        x = idx
        x_mean = x.mean()
        y_mean = w.mean()
        cov = np.sum((x - x_mean) * (w - y_mean))
        varx = np.sum((x - x_mean) ** 2)
        if varx == 0:
            continue
        slope = cov / varx
        intercept = y_mean - slope * x_mean
        y_hat = intercept + slope * x
        ss_res = np.sum((w - y_hat) ** 2)
        ss_tot = np.sum((w - y_mean) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        slopes[i] = slope
        r2s[i] = r2

    return pd.Series(slopes, index=close.index), pd.Series(r2s, index=close.index)


def compute_features(
    bars: pd.DataFrame,
    chop_len: int,
    adx_len: int,
    er_len: int,
    atr_len: int,
    rvol_len: int,
    rsi_len: int,
    trend_len: int,
) -> pd.DataFrame:
    df = bars.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    df = df.reset_index(drop=True)

    close = df["close"].astype(float)

    feat = pd.DataFrame({"ts": df["ts"].values})
    feat["CHOP"] = _choppiness(df, chop_len)
    feat["ADX"] = _adx(df, adx_len)
    feat["ER"] = _er_kaufman(close, er_len)
    feat["ATR_PCT"] = _atr_pct(df, atr_len)
    feat["RVOL_PCT"] = _rvol_pct(df, rvol_len)
    feat["RSI"] = _rsi(close, rsi_len)
    slope, r2 = _trend_slope_and_r2(close, trend_len)
    feat["TREND_SLOPE"] = slope
    feat["TREND_R2"] = r2

    return feat


# -----------------------------
# Interactive window marking
# -----------------------------

@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp
    label: str


def _read_bars_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"parquet missing column: {c}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return df


def _read_equity(run_dir: str) -> pd.DataFrame:
    p = Path(run_dir) / "equity.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (run_dir must contain equity.parquet)")
    eq = pd.read_parquet(p)
    if "ts" not in eq.columns or "equity" not in eq.columns:
        raise ValueError("equity.parquet must have columns: ts, equity")
    eq["ts"] = pd.to_datetime(eq["ts"], utc=True, errors="coerce")
    eq = eq.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return eq


def _nearest_ts(ts_series: pd.Series, target: pd.Timestamp) -> pd.Timestamp:
    # ensure tz-aware UTC
    t = pd.to_datetime(target, utc=True)
    s = pd.to_datetime(ts_series, utc=True)
    # use numpy searchsorted
    arr = s.values.astype("datetime64[ns]")
    x = np.datetime64(t.to_datetime64())
    i = np.searchsorted(arr, x)
    if i <= 0:
        return s.iloc[0]
    if i >= len(s):
        return s.iloc[-1]
    prev = s.iloc[i - 1]
    nxt = s.iloc[i]
    return prev if abs((prev - t).total_seconds()) <= abs((nxt - t).total_seconds()) else nxt


def interactive_mark_windows(eq: pd.DataFrame, out_windows_csv: str) -> List[Window]:
    """
    Click pairs: start then end for each window.
    Close the plot window when done.
    """
    out_path = Path(out_windows_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(eq["ts"].values, eq["equity"].values)
    ax.set_title("Click START then END for each positive phase (close window when done)")
    ax.set_xlabel("time")
    ax.set_ylabel("equity")

    print("\nINTERACTIVE MARKING")
    print(" - Click START then END for each window you marked as positive.")
    print(" - Repeat for multiple windows.")
    print(" - When finished: close the plot window.\n")

    pts = plt.ginput(n=-1, timeout=0)  # list of (x,y) with x in Matplotlib date float
    plt.close(fig)

    if len(pts) < 2:
        print("No windows captured (need at least 2 clicks).")
        return []

    # Convert x (matplotlib date num) to timestamps
    xs = [pd.to_datetime(mpl_num, unit="D", origin="1899-12-30", utc=True) for mpl_num, _ in pts]

    windows: List[Window] = []
    k = 0
    for i in range(0, len(xs) - 1, 2):
        a = _nearest_ts(eq["ts"], xs[i])
        b = _nearest_ts(eq["ts"], xs[i + 1])
        start = min(a, b)
        end = max(a, b)
        k += 1
        windows.append(Window(start=start, end=end, label=f"w{k:02d}"))

    out_df = pd.DataFrame(
        [{"start": w.start.isoformat().replace("+00:00", "Z"),
          "end": w.end.isoformat().replace("+00:00", "Z"),
          "label": w.label} for w in windows]
    )
    out_df.to_csv(out_path, index=False)
    print(f"WROTE {out_path}  (windows={len(windows)})")
    return windows


def load_windows_csv(path: str) -> List[Window]:
    df = pd.read_csv(path)
    if not {"start", "end"}.issubset(df.columns):
        raise ValueError("windows csv must have columns: start,end,(optional)label")
    if "label" not in df.columns:
        df["label"] = [f"w{i+1:02d}" for i in range(len(df))]
    df["start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    out = []
    for _, r in df.iterrows():
        out.append(Window(start=r["start"], end=r["end"], label=str(r["label"])))
    return out


# -----------------------------
# Window feature summary
# -----------------------------

def summarize_window(feat: pd.DataFrame, w: Window) -> dict:
    # FIX for your error: make sure BOTH sides are tz-aware UTC
    ts = pd.to_datetime(feat["ts"], utc=True)
    start = pd.to_datetime(w.start, utc=True)
    end = pd.to_datetime(w.end, utc=True)

    m = (ts >= start) & (ts < end)
    sub = feat.loc[m].copy()

    row = {"label": w.label, "start": start.isoformat(), "end": end.isoformat(), "rows": int(len(sub))}
    for c in ["CHOP", "ADX", "ER", "ATR_PCT", "RVOL_PCT", "RSI", "TREND_SLOPE", "TREND_R2"]:
        s = pd.to_numeric(sub[c], errors="coerce").dropna()
        if len(s) == 0:
            row[f"{c}_mean"] = np.nan
            row[f"{c}_p50"] = np.nan
            continue
        row[f"{c}_mean"] = float(s.mean())
        row[f"{c}_p50"] = float(s.quantile(0.5))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="OHLCV parquet with ts,open,high,low,close,(optional)volume")
    ap.add_argument("--run-dir", required=True, help="run dir containing equity.parquet")
    ap.add_argument("--windows-csv", default="data/marked_windows.csv", help="where to read/write marked windows")
    ap.add_argument("--out", default="data/marked_windows_features.csv", help="output features per window")

    ap.add_argument("--chop-len", type=int, default=14)
    ap.add_argument("--adx-len", type=int, default=14)
    ap.add_argument("--er-len", type=int, default=120)
    ap.add_argument("--atr-len", type=int, default=14)
    ap.add_argument("--rvol-len", type=int, default=20)
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--trend-len", type=int, default=240)

    ap.add_argument("--no-ui", action="store_true", help="do not open interactive plot; requires windows-csv to exist")
    args = ap.parse_args()

    bars = _read_bars_parquet(args.parquet)
    eq = _read_equity(args.run_dir)

    windows_path = Path(args.windows_csv)

    if not args.no_ui:
        # create/overwrite windows via UI
        windows = interactive_mark_windows(eq, str(windows_path))
    else:
        if not windows_path.exists():
            raise FileNotFoundError(f"--no-ui set but windows file missing: {windows_path}")
        windows = load_windows_csv(str(windows_path))

    if not windows:
        print("No windows -> nothing to analyze.")
        return

    feat = compute_features(
        bars=bars,
        chop_len=args.chop_len,
        adx_len=args.adx_len,
        er_len=args.er_len,
        atr_len=args.atr_len,
        rvol_len=args.rvol_len,
        rsi_len=args.rsi_len,
        trend_len=args.trend_len,
    )

    rows = [summarize_window(feat, w) for w in windows]
    out_df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"WROTE {out_path}  (rows={len(out_df)})")
    print("\nWINDOWS (just dates):")
    for w in windows:
        print(f"{w.start.isoformat()}  ..  {w.end.isoformat()}  ({w.label})")


if __name__ == "__main__":
    main()
