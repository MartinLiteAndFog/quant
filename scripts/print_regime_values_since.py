# scripts/print_regime_values_since.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def choppiness(df: pd.DataFrame, n: int) -> pd.Series:
    tr = true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = df["high"].astype(float).rolling(n, min_periods=n).max()
    ll = df["low"].astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))


def wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()


def adx(df: pd.DataFrame, n: int) -> pd.Series:
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


def describe(name: str, s: pd.Series) -> None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        print(f"{name}: (no data)")
        return
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    print(
        f"{name}: n={len(s)}  mean={s.mean():.2f}  "
        f"p05={q[0.05]:.2f} p25={q[0.25]:.2f} p50={q[0.5]:.2f} p75={q[0.75]:.2f} p95={q[0.95]:.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--run-dir", default=None, help="optional: data/runs/<run_id> to sample entry points")
    ap.add_argument("--since", default="2026-01-10", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--chop-len", type=int, default=14)
    ap.add_argument("--adx-len", type=int, default=14)
    ap.add_argument("--chop-on", type=float, default=54.0)
    ap.add_argument("--chop-off", type=float, default=48.0)  # not used for stats, just printed
    ap.add_argument("--adx-on", type=float, default=18.0)
    ap.add_argument("--adx-off", type=float, default=25.0)
    args = ap.parse_args()

    bars = read_ohlcv_parquet(args.parquet)
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    since_ts = pd.Timestamp(args.since, tz="UTC")

    bars = bars[bars["ts"] >= since_ts].reset_index(drop=True)
    print(f"Window: {since_ts} .. {bars['ts'].max()}  rows={len(bars)}")
    print(f"Thresholds (current winner CHOP): chop_on={args.chop_on} chop_off={args.chop_off} | adx_on={args.adx_on} adx_off={args.adx_off}")

    c = choppiness(bars, args.chop_len)
    a = adx(bars, args.adx_len)

    describe("CHOP", c)
    describe("ADX", a)

    # optional: sample only at entry timestamps (more relevant)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        events_p = run_dir / "events.parquet"
        if events_p.exists():
            ev = pd.read_parquet(events_p)
            ev["ts"] = pd.to_datetime(ev["ts"], utc=True, errors="coerce")
            ev = ev[(ev["event"] == "entry") & (ev["ts"] >= since_ts)].dropna(subset=["ts"]).copy()
            if len(ev) == 0:
                print("No entries in this window.")
                return

            # align by exact ts
            idx = pd.DatetimeIndex(bars["ts"])
            c_s = pd.Series(c.values, index=idx)
            a_s = pd.Series(a.values, index=idx)
            common = ev["ts"].isin(idx)
            ev = ev[common].copy()

            ev["chop"] = ev["ts"].map(c_s.to_dict())
            ev["adx"] = ev["ts"].map(a_s.to_dict())

            print(f"\nEntries since {args.since}: {len(ev)}")
            describe("CHOP@entry", ev["chop"])
            describe("ADX@entry", ev["adx"])
        else:
            print(f"run-dir given but missing {events_p}")


if __name__ == "__main__":
    main()
