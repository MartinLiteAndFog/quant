from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant.features.renko import renko_from_close
from quant.strategies.imba import ImbaParams, make_signals_from_ohlcv


def _read_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    # We only need close for renko generation
    if "close" not in df.columns:
        raise ValueError("parquet missing 'close' column")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    return df[["ts", "close"]].copy()


def _renko_to_ohlc(bricks: pd.DataFrame) -> pd.DataFrame:
    """
    Convert renko bricks (ts, dir, open, close) to OHLC bars compatible with IMBA:
      open = brick open
      close = brick close
      high = max(open, close)
      low  = min(open, close)
    """
    b = bricks.copy()
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    b["open"] = pd.to_numeric(b["open"], errors="coerce")
    b["close"] = pd.to_numeric(b["close"], errors="coerce")
    b = b.dropna(subset=["ts", "open", "close"]).sort_values("ts").reset_index(drop=True)

    hi = b[["open", "close"]].max(axis=1)
    lo = b[["open", "close"]].min(axis=1)

    out = pd.DataFrame(
        {
            "ts": b["ts"].values,
            "open": b["open"].values,
            "high": hi.values,
            "low": lo.values,
            "close": b["close"].values,
        }
    )

    # Important: IMBA does rolling(high/low). Duplicate timestamps are allowed in Renko.
    # To keep deterministic order and avoid duplicate-ts reindex headaches inside IMBA,
    # we make ts unique by adding nanoseconds within same ts group.
    if len(out) > 1:
        dup = out["ts"].duplicated(keep=False)
        if dup.any():
            grp = out["ts"].astype("int64")
            idx_in_grp = out.groupby(grp).cumcount()
            out["ts"] = out["ts"] + pd.to_timedelta(idx_in_grp, unit="ns")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Input OHLCV parquet (we use ts+close)")
    ap.add_argument("--box", type=float, required=True, help="Renko box size")
    ap.add_argument("--lookback", type=int, default=240, help="IMBA lookback (lb), e.g. 240")
    ap.add_argument("--sl-abs", type=float, default=1.5, help="Fixed absolute SL distance (quote currency)")
    ap.add_argument("--start", type=str, default=None, help="Optional UTC ISO start, e.g. 2026-01-26T00:00:00Z")
    ap.add_argument("--out", required=True, help="Output jsonl path")
    args = ap.parse_args()

    df = _read_ohlcv_parquet(args.parquet)
    if args.start:
        start = pd.to_datetime(args.start, utc=True)
        df = df[df["ts"] >= start].reset_index(drop=True)

    bricks = renko_from_close(df, box=float(args.box))
    if len(bricks) == 0:
        raise ValueError("No renko bricks formed (data too flat or box too large).")

    renko_ohlc = _renko_to_ohlc(bricks)

    params = ImbaParams(
        lookback=int(args.lookback),
        start_ts=None,
        fixed_sl_abs=float(args.sl_abs),
    )

    n, sig = make_signals_from_ohlcv(
        df_ohlcv=renko_ohlc,
        params=params,
        out_jsonl=Path(args.out),
    )

    print(f"WROTE {args.out}")
    print(f"RENKO bricks: {len(bricks)}   range: {bricks['ts'].min()} -> {bricks['ts'].max()}")
    print(f"IMBA signals written: {n}")
    if len(sig):
        print(f"Signals range: {sig['ts'].min()} -> {sig['ts'].max()}")


if __name__ == "__main__":
    main()
