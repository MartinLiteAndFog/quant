# scripts/print_regime_window_stats.py
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np

def _read_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index":"ts"})
        else:
            raise ValueError("parquet missing ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df

def _true_range(df: pd.DataFrame) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    pc = c.shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def _chop(df: pd.DataFrame, n: int) -> pd.Series:
    tr = _true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = df["high"].astype(float).rolling(n, min_periods=n).max()
    ll = df["low"].astype(float).rolling(n, min_periods=n).min()
    denom = (hh-ll).replace(0.0, np.nan)
    return 100.0 * np.log10(sum_tr/denom) / np.log10(float(n))

def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0/float(n), adjust=False).mean()

def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    n=int(n)
    h=df["high"].astype(float); l=df["low"].astype(float)
    up=h.diff()
    down=-l.diff()
    dm_p=pd.Series(np.where((up>down)&(up>0), up, 0.0), index=df.index)
    dm_m=pd.Series(np.where((down>up)&(down>0), down, 0.0), index=df.index)
    tr=_true_range(df)
    atr=_wilder(tr,n)
    sm_p=_wilder(dm_p,n)
    sm_m=_wilder(dm_m,n)
    di_p=100.0*(sm_p/atr.replace(0.0,np.nan))
    di_m=100.0*(sm_m/atr.replace(0.0,np.nan))
    dx=100.0*(di_p-di_m).abs()/(di_p+di_m).replace(0.0,np.nan)
    return _wilder(dx,n)

def _er(df: pd.DataFrame, n: int) -> pd.Series:
    n=int(n)
    c=df["close"].astype(float)
    change=(c - c.shift(n)).abs()
    vol=(c.diff().abs()).rolling(n, min_periods=n).sum()
    return (change/vol.replace(0.0,np.nan))

def _summary(x: pd.Series, name: str) -> dict:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if len(s)==0:
        return {"name":name, "n":0}
    q = s.quantile([0.05,0.25,0.50,0.75,0.95]).to_dict()
    return {
        "name": name,
        "n": int(len(s)),
        "mean": float(s.mean()),
        "p05": float(q.get(0.05)),
        "p25": float(q.get(0.25)),
        "p50": float(q.get(0.50)),
        "p75": float(q.get(0.75)),
        "p95": float(q.get(0.95)),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--start", required=True, help="ISO timestamp, e.g. 2026-01-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="ISO timestamp")
    ap.add_argument("--chop-len", type=int, default=14)
    ap.add_argument("--adx-len", type=int, default=14)
    ap.add_argument("--er-len", type=int, default=120)
    args=ap.parse_args()

    df=_read_ohlcv_parquet(args.parquet)
    start=pd.to_datetime(args.start, utc=True)
    end=pd.to_datetime(args.end, utc=True)
    w=df[(df["ts"]>=start)&(df["ts"]<end)].copy()
    print(f"WINDOW {start}..{end} rows={len(w)}")

    out=[]
    out.append(_summary(_chop(w,args.chop_len), f"CHOP({args.chop_len})"))
    out.append(_summary(_adx(w,args.adx_len), f"ADX({args.adx_len})"))
    out.append(_summary(_er(w,args.er_len), f"ER({args.er_len})"))

    for r in out:
        if r.get("n",0)==0:
            print(f"{r['name']}: n=0")
        else:
            print(f"{r['name']}: n={r['n']} mean={r['mean']:.4f} p05={r['p05']:.4f} p25={r['p25']:.4f} p50={r['p50']:.4f} p75={r['p75']:.4f} p95={r['p95']:.4f}")

if __name__=="__main__":
    main()
