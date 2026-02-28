#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet/CSV file")
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--limit", type=int, default=0, help="optional row limit for CSV")
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise SystemExit(f"file not found: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p, nrows=(args.limit or None))
    else:
        raise SystemExit("unsupported file type (use .parquet or .csv)")

    if args.ts_col not in df.columns:
        raise SystemExit(f"missing ts column: {args.ts_col} ; cols={list(df.columns)}")

    s = df[args.ts_col]
    print("rows:", len(df))
    print("ts_dtype:", s.dtype)
    print("ts_nulls:", int(s.isna().sum()))

    # sortable?
    sortable = True
    try:
        _ = s.sort_values(kind="mergesort")
    except Exception as e:
        sortable = False
        print("sortable: no", "err:", repr(e))
    else:
        print("sortable: yes")

    # normalize to datetime if possible
    s_dt = None
    if pd.api.types.is_datetime64_any_dtype(s):
        s_dt = s
    else:
        # try parse to datetime; if fails, keep as-is
        try:
            s_dt = pd.to_datetime(s, utc=True, errors="raise")
        except Exception:
            s_dt = None

    s_use = s_dt if s_dt is not None else s
    dup = int(s_use.duplicated().sum())
    print("duplicate_ts_rows:", dup)

    mono = bool(s_use.is_monotonic_increasing)
    print("is_monotonic_increasing:", mono)

    if dup == 0 and mono:
        print("no_write_needed: already sorted and unique")
    else:
        print("needs_fix: sort+dedup recommended")
        # show a tiny sample of problems
        if dup > 0:
            dups = df.loc[s_use.duplicated(keep=False), [args.ts_col]].head(10)
            print("\nfirst_duplicate_samples:")
            print(dups.to_string(index=False))


if __name__ == "__main__":
    main()