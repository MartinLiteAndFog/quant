# scripts/make_signals_imba_renko.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant.features.renko import renko_from_close
from quant.strategies.imba import ImbaParams, make_signals_from_ohlcv


def _read_ts_close_from_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # ts can be a column or datetime index
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            # common cases: index name is 'ts' or 'index'
            if "ts" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    if "close" not in df.columns:
        raise ValueError("parquet missing 'close' column")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    return df[["ts", "close"]].copy()


def _renko_bricks_to_ohlc(bricks: pd.DataFrame) -> pd.DataFrame:
    """
    bricks: columns usually include ts, dir, open, close (from renko_from_close)
    Convert to OHLC bars for IMBA:
      open = brick open
      close = brick close
      high = max(open, close)
      low  = min(open, close)

    Important: Renko can contain duplicate timestamps.
    We make timestamps unique by adding ns offsets within same ts group
    so any rolling logic stays deterministic.
    """
    b = bricks.copy()

    if "ts" not in b.columns:
        raise ValueError("renko bricks missing 'ts'")
    if "open" not in b.columns or "close" not in b.columns:
        raise ValueError("renko bricks missing 'open'/'close' columns")

    b["ts"] = pd.to_datetime(b["ts"], utc=True, errors="coerce")
    b["open"] = pd.to_numeric(b["open"], errors="coerce")
    b["close"] = pd.to_numeric(b["close"], errors="coerce")
    b = b.dropna(subset=["ts", "open", "close"]).sort_values("ts").reset_index(drop=True)

    out = pd.DataFrame(
        {
            "ts": b["ts"].values,
            "open": b["open"].values,
            "close": b["close"].values,
        }
    )
    out["high"] = out[["open", "close"]].max(axis=1)
    out["low"] = out[["open", "close"]].min(axis=1)

    # Make ts unique if duplicates exist (ns offsets)
    if len(out) > 1:
        dup = out["ts"].duplicated(keep=False)
        if dup.any():
            grp = out["ts"].astype("int64")
            idx_in_grp = out.groupby(grp).cumcount()
            out["ts"] = out["ts"] + pd.to_timedelta(idx_in_grp, unit="ns")

    return out[["ts", "open", "high", "low", "close"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Input OHLCV parquet (we use ts+close)")
    ap.add_argument("--box", type=float, required=True, help="Renko box size")
    ap.add_argument("--lookback", type=int, default=240, help="IMBA lookback")
    ap.add_argument("--sl-abs", type=float, default=1.5, help="IMBA fixed SL abs (quote currency)")
    ap.add_argument("--start", type=str, default="", help="Optional UTC ISO start, e.g. 2021-09-24T10:00:00Z")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    df = _read_ts_close_from_parquet(args.parquet)

    if args.start:
        start = pd.to_datetime(args.start, utc=True)
        df = df[df["ts"] >= start].reset_index(drop=True)

    bricks = renko_from_close(df, box=float(args.box))
    if bricks is None or len(bricks) == 0:
        raise ValueError("No renko bricks formed (data too flat or box too large).")

    renko_ohlc = _renko_bricks_to_ohlc(bricks)

    params = ImbaParams(
        lookback=int(args.lookback),
        start_ts=None,
        fixed_sl_abs=float(args.sl_abs),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n, sig = make_signals_from_ohlcv(
        df_ohlcv=renko_ohlc,
        params=params,
        out_jsonl=out,
    )

    print(f"WROTE {out}")
    print(f"RENKO bricks: {len(bricks)}   range: {bricks['ts'].min()} -> {bricks['ts'].max()}")
    print(f"IMBA signals written: {n}")
    if hasattr(sig, "__len__") and len(sig):
        try:
            print(f"Signals range: {sig['ts'].min()} -> {sig['ts'].max()}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
