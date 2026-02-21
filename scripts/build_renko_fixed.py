# scripts/build_renko_fixed.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def read_ohlcv_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_parquet(p)

    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex) and df.index.name == "ts":
            df = df.reset_index()
        else:
            raise ValueError("Expected 'ts' column or DatetimeIndex named 'ts'")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def build_renko_fixed(
    df: pd.DataFrame,
    box: float,
    gap_sec: float = 180.0,
) -> pd.DataFrame:
    """
    Fixed-box Renko from OHLCV. Uses candle close as price stream.
    Key rule: if time gap > gap_sec, reset anchor to current close (no bricks across gap).
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "dir", "src_ts"])

    ts = df["ts"]
    close = df["close"].astype(float)

    bricks: List[Dict] = []

    last_ts = ts.iloc[0]
    last_close = float(close.iloc[0])

    for i in range(1, len(df)):
        t = ts.iloc[i]
        px = float(close.iloc[i])

        dt = (t - last_ts).total_seconds()
        if dt > gap_sec:
            # session break: do NOT create bricks that jump across missing tape
            last_ts = t
            last_close = px
            continue

        move = px - last_close

        # Emit as many bricks as needed
        while abs(move) >= box:
            d = 1 if move > 0 else -1
            nxt = last_close + d * box
            o = last_close
            c = nxt
            hi = max(o, c)
            lo = min(o, c)

            bricks.append(
                {
                    "ts": t,          # assign brick timestamp to current candle ts
                    "open": o,
                    "high": hi,
                    "low": lo,
                    "close": c,
                    "dir": d,         # +1 up brick, -1 down brick
                    "src_ts": t,      # keep explicit source ts (same as ts here, but useful later)
                }
            )
            last_close = nxt
            move = px - last_close

        last_ts = t

    out = pd.DataFrame(bricks)
    if len(out) == 0:
        return out

    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="OHLCV parquet with ts/open/high/low/close")
    ap.add_argument("--out", required=True, help="Output renko parquet")
    ap.add_argument("--box", type=float, required=True, help="Renko box size, e.g. 0.1")
    ap.add_argument("--gap-sec", type=float, default=180.0, help="Session break threshold in seconds (default 180)")
    ap.add_argument("--from-ts", default=None, help="Optional start ts, e.g. 2023-12-28T08:00:00Z")
    args = ap.parse_args()

    df = read_ohlcv_parquet(args.inp)

    if args.from_ts:
        t0 = pd.to_datetime(args.from_ts, utc=True)
        df = df[df["ts"] >= t0].reset_index(drop=True)

    renko = build_renko_fixed(df, box=float(args.box), gap_sec=float(args.gap_sec))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    renko.to_parquet(outp, index=False)

    print("in rows:", len(df))
    print("renko bricks:", len(renko))
    if len(renko):
        print("renko range:", renko["ts"].iloc[0], "->", renko["ts"].iloc[-1])
        print("last close:", float(renko["close"].iloc[-1]))
    print("wrote:", outp)


if __name__ == "__main__":
    main()
