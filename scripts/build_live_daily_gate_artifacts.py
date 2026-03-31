#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from quant.execution.daily_gate_artifacts import write_daily_gate_artifacts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build live-ready daily gate CSV artifacts from a PC gate file or predictions parquet."
    )
    ap.add_argument(
        "--input-path",
        default=None,
        help="Source CSV/parquet. If omitted, uses PC_PREDICTIONS_PARQUET or /data/live/gate_conf/predictions.parquet.",
    )
    ap.add_argument(
        "--out-on-csv",
        default=None,
        help="Output CSV for countertrend/ON gate. Defaults to GATE_DAILY_PATH or /data/live/gate_conf/gate_daily.csv.",
    )
    ap.add_argument(
        "--out-off-csv",
        default=None,
        help="Output CSV for trend/OFF gate. Defaults to GATE_DAILY_OFF_PATH or /data/live/gate_conf/gate_daily_off.csv.",
    )
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--on-source-col", default="gate_base_2of3")
    ap.add_argument("--off-source-col", default=None)
    args = ap.parse_args()

    import os

    input_path = (
        args.input_path
        or os.getenv("PC_PREDICTIONS_PARQUET")
        or "/data/live/gate_conf/predictions.parquet"
    )
    out_on_csv = (
        args.out_on_csv
        or os.getenv("GATE_DAILY_PATH")
        or "/data/live/gate_conf/gate_daily.csv"
    )
    out_off_csv = (
        args.out_off_csv
        or os.getenv("GATE_DAILY_OFF_PATH")
        or "/data/live/gate_conf/gate_daily_off.csv"
    )

    on_df, off_df = write_daily_gate_artifacts(
        input_path=input_path,
        out_on_path=out_on_csv,
        out_off_path=out_off_csv,
        ts_col=args.ts_col,
        on_source_col=args.on_source_col,
        off_source_col=args.off_source_col,
    )

    print(
        "INFO wrote",
        Path(out_on_csv),
        "rows=",
        len(on_df),
        "col=gate_on_2of3",
    )
    print(
        "INFO wrote",
        Path(out_off_csv),
        "rows=",
        len(off_df),
        "col=gate_off_2of3",
    )


if __name__ == "__main__":
    main()
