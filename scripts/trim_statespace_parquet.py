from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/Users/martinpeter/Desktop/quant/data/state_space/solusdt_ohlcv_state_space_v01.parquet"
)


def infer_unix_unit(values: pd.Series) -> str:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    v = np.abs(v[np.isfinite(v)])
    if v.size == 0:
        return "s"
    scale = float(np.nanmedian(v))
    if scale >= 1e17:
        return "ns"
    if scale >= 1e14:
        return "us"
    if scale >= 1e11:
        return "ms"
    return "s"


def parse_ts_to_utc(ts: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(ts.dtype):
        unit = infer_unix_unit(ts)
        print(f"ts numeric detected -> unix unit heuristic: {unit}")
        return pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(ts, utc=True, errors="coerce")


def prepare_dataframe(df: pd.DataFrame, ts_col: str = "ts", months: int | None = None) -> pd.DataFrame:
    if ts_col not in df.columns:
        raise ValueError(f"missing required column: {ts_col}")

    out = df.copy()
    out["_row"] = np.arange(len(out), dtype=np.int64)
    out["ts_dt"] = parse_ts_to_utc(out[ts_col])

    nat_count = int(out["ts_dt"].isna().sum())
    print(f"ts dtype: {out[ts_col].dtype}")
    print(f"ts_dt NaT present: {nat_count > 0} (count={nat_count})")
    print(f"ts_dt min: {out['ts_dt'].min()} | max: {out['ts_dt'].max()}")
    print("ts_dt head:", out["ts_dt"].head(5).tolist())
    print("ts_dt tail:", out["ts_dt"].tail(5).tolist())

    # Stable sort by parsed timestamp, then original row order.
    out = out.sort_values(["ts_dt", "_row"], kind="mergesort").reset_index(drop=True)

    before = len(out)
    out = out.drop_duplicates(subset=["ts_dt"], keep="first").reset_index(drop=True)
    after = len(out)
    print(f"rows before dedup: {before}, after dedup: {after}, dupes removed: {before - after}")

    if months is not None:
        if months <= 0:
            raise ValueError("--months must be > 0")
        ts_max = out["ts_dt"].max()
        if pd.isna(ts_max):
            raise ValueError("cannot apply --months because ts_dt max is NaT")
        cutoff = ts_max - pd.DateOffset(months=int(months))
        before_cut = len(out)
        out = out[out["ts_dt"] >= cutoff].copy().reset_index(drop=True)
        print(f"truncate months={months}: cutoff={cutoff}, rows before={before_cut}, after={len(out)}")

    out = out.drop(columns=["_row"])
    return out


def output_path_for(input_path: Path, months: int | None) -> Path:
    stem = input_path.stem
    if months is None:
        name = f"{stem}_SORTED_DEDUP.parquet"
    else:
        name = f"{stem}_LAST{int(months)}M.parquet"
    return input_path.parent / name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--months", type=int, default=None)
    ap.add_argument("--ts-col", default="ts")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    cleaned = prepare_dataframe(df, ts_col=args.ts_col, months=args.months)

    out_path = output_path_for(input_path, args.months)
    cleaned.to_parquet(out_path, index=False)
    print(f"wrote: {out_path}")
    print(f"final rows: {len(cleaned)}")


if __name__ == "__main__":
    main()
