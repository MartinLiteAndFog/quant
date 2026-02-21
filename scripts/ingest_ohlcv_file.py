#!/usr/bin/env python3
"""
Ingest a local OHLCV file (.csv / .csv.gz / .zip containing csv) into the quant
canonical schema and write a parquet file.

Canonical schema:
  ts (UTC), open, high, low, close, volume

Examples:
  python scripts/ingest_ohlcv_file.py \
    --in "/path/to/SOLUSDT-1m-2025-02_to_2026-02.csv.gz" \
    --out "data/raw/exchange=local/symbol=SOL-USDT/timeframe=1m/SOL-USDT_1m_20250207T000000Z.parquet"
"""
import argparse
import io
import os
import zipfile
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd


def _read_any_csv(path: str) -> pd.DataFrame:
    # .zip: take first .csv in archive
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("ZIP contains no .csv files")
            # pick first csv (or you can improve with --member)
            name = sorted(csv_names)[0]
            with zf.open(name) as f:
                raw = f.read()
            return pd.read_csv(io.BytesIO(raw))

    # .gz / .csv / others that pandas can infer
    return pd.read_csv(path, compression="infer")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to normalize common exchange CSV layouts.

    Supported patterns:
      1) Already canonical: ts, open, high, low, close, volume
      2) Binance-vision style:
         open_time, open, high, low, close, volume, close_time, ...
      3) Generic:
         timestamp or date columns + O/H/L/C/V columns with various names
    """
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    lower_map = {c.lower(): c for c in df.columns}

    # Helper: find first matching column name
    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in lower_map:
                return lower_map[n.lower()]
        return None

    # 1) canonical
    if set(["ts", "open", "high", "low", "close", "volume"]).issubset(set(df.columns)):
        out = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    else:
        # time column candidates
        ts_col = pick("ts", "timestamp", "time", "date", "open_time", "open time", "datetime")

        if ts_col is None:
            raise ValueError(f"Could not find a timestamp column in: {list(df.columns)}")

        # price/volume columns candidates
        o = pick("open", "o")
        h = pick("high", "h")
        l = pick("low", "l")
        c = pick("close", "c")
        v = pick("volume", "vol", "v")

        if not all([o, h, l, c, v]):
            raise ValueError(
                "Could not find O/H/L/C/V columns. "
                f"Columns present: {list(df.columns)}"
            )

        out = df[[ts_col, o, h, l, c, v]].copy()
        out.columns = ["ts", "open", "high", "low", "close", "volume"]

    # --- parse ts ---
    # If numeric: assume ms or s. Heuristic: ms are typically > 1e12
    if pd.api.types.is_numeric_dtype(out["ts"]):
        s = out["ts"].astype("int64")
        unit = "ms" if s.median() > 10**12 else "s"
        out["ts"] = pd.to_datetime(out["ts"], unit=unit, utc=True)
    else:
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")

    out = out.dropna(subset=["ts"])

    # ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.sort_values("ts")
    out = out.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input .csv/.csv.gz/.zip")
    ap.add_argument("--out", dest="out_path", required=True, help="Output .parquet path")
    args = ap.parse_args()

    df_raw = _read_any_csv(args.in_path)
    df = _normalize_ohlcv(df_raw)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df.to_parquet(args.out_path, index=False)

    print(f"ingested rows={len(df)}")
    print(f"start={df['ts'].iloc[0]}  end={df['ts'].iloc[-1]}")
    print(f"wrote: {args.out_path}")


if __name__ == "__main__":
    main()
