from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant.state_space.config import StateSpaceConfig
from quant.state_space.pipeline import compute_state_space


def _load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .parquet or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v0.1 3-axis market state space pipeline")
    parser.add_argument("--input", required=True, help="Path to input parquet/csv")
    parser.add_argument("--output", required=True, help="Path to output parquet")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    df = _load_input(in_path)
    out = compute_state_space(df, StateSpaceConfig())
    out.to_parquet(out_path, index=False)

    basins = out.attrs.get("basins", [])
    if basins:
        basin_path = out_path.with_name(out_path.stem + "_basins.csv")
        pd.DataFrame(basins).to_csv(basin_path, index=False)


if __name__ == "__main__":
    main()
