from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant.state_space_transitions.config import TransitionConfig
from quant.state_space_transitions.pipeline import run_pipeline


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input format: {path.suffix}")


def _print_summary(out_dir: Path) -> None:
    vox = pd.read_parquet(out_dir / "voxel_stats.parquet")
    basins = pd.read_parquet(out_dir / "basin_stats.parquet")
    dfv = pd.read_parquet(out_dir / "df_with_voxels.parquet")

    print("\n=== Current Voxel Stats (last row) ===")
    if dfv.empty:
        print("no rows")
    else:
        last = dfv.iloc[-1]
        vid = int(last["voxel_id"])
        row = vox[vox["voxel_id"] == vid]
        if row.empty:
            print(f"last voxel_id={vid} not found in voxel_stats")
        else:
            print(row.to_string(index=False))

    print("\n=== Top 10 Voxels by occ_eff ===")
    if vox.empty:
        print("no voxel stats")
    else:
        cols = ["voxel_id", "occ_eff", "p_self", "holding_time", "entropy", "speed"]
        print(vox.nlargest(10, "occ_eff")[cols].to_string(index=False))

    print("\n=== Top Basins by occ_share ===")
    if basins.empty:
        print("no basin stats")
    else:
        cols = ["basin_id", "occ_share", "basin_persistence"]
        print(basins.nlargest(10, "occ_share")[cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-bins", type=int, default=40)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--n-min-voxel", type=int, default=20)
    ap.add_argument("--decay-halflife-bars", type=int, default=0)
    ap.add_argument("--basin-k", type=int, default=30)
    ap.add_argument("--lookback-days", type=int, default=None)
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--x-col", default="X_raw")
    ap.add_argument("--y-col", default="Y_res")
    ap.add_argument("--z-col", default="Z_res")
    args = ap.parse_args()

    input_path = Path(args.input)
    df = _read_input(input_path)

    rename_map = {
        args.ts_col: "ts",
        args.x_col: "X_raw",
        args.y_col: "Y_res",
        args.z_col: "Z_res",
    }
    missing = [k for k in rename_map.keys() if k not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in input: {missing}")
    df = df.rename(columns=rename_map)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")

    if args.lookback_days is not None and args.lookback_days > 0 and not df.empty:
        cutoff = df["ts"].max() - pd.Timedelta(days=int(args.lookback_days))
        df = df[df["ts"] >= cutoff].copy()

    cfg = TransitionConfig(
        n_bins=int(args.n_bins),
        bin_method="quantile",
        n_min_voxel=int(args.n_min_voxel),
        topk=int(args.topk),
        laplace_alpha=float(args.alpha),
        decay_halflife_bars=int(args.decay_halflife_bars),
        axes=["X_raw", "Y_res", "Z_res"],
        run_id=str(args.run_id),
    )

    out_dir = Path("data") / "runs" / cfg.run_id / "transitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(df=df, cfg=cfg, out_dir=out_dir, basin_k=int(args.basin_k))

    print("WROTE:", out_dir)
    print("COUNTS:", result)
    _print_summary(out_dir)


if __name__ == "__main__":
    main()
