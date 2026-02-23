#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from quant.regime import RegimeService, RegimeStore


def _load_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest gate/regime rows into SQLite regime store")
    ap.add_argument("--input", required=True, help="CSV/parquet with ts and gate column")
    ap.add_argument("--symbol", default="SOL-USDT")
    ap.add_argument("--db-path", default=None, help="Override REGIME_DB_PATH")
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--gate-col", default="gate_on")
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--confidence-col", default=None)
    ap.add_argument("--reason-code", default="ingest_gate_df")
    ap.add_argument("--model-version", default="regime-v1")
    ap.add_argument("--snapshot-id", default=None)
    ap.add_argument("--snapshot-ts", default=None, help="fitted timestamp for threshold snapshot (ISO)")
    ap.add_argument("--snapshot-params-json", default=None, help="Optional JSON string with threshold params")
    ap.add_argument("--on-state", default="trend")
    ap.add_argument("--off-state", default="countertrend")
    ap.add_argument("--source-name", default="gate_file")
    ap.add_argument("--missing-bars", type=int, default=0)
    ap.add_argument("--stale-age-sec", type=float, default=0.0)
    ap.add_argument("--fallback-used", action="store_true")
    args = ap.parse_args()

    df = _load_frame(args.input)
    store = RegimeStore(db_path=args.db_path)
    svc = RegimeService(store)

    if args.snapshot_id:
        params = {}
        if args.snapshot_params_json:
            params = json.loads(args.snapshot_params_json)
        store.insert_threshold_snapshot(
            snapshot_id=str(args.snapshot_id),
            symbol=str(args.symbol),
            fitted_at=str(args.snapshot_ts or pd.Timestamp.utcnow().isoformat()),
            params=params,
            model_version=str(args.model_version),
        )

    inserted = svc.ingest_gate_dataframe(
        df=df,
        symbol=str(args.symbol),
        gate_col=str(args.gate_col),
        ts_col=str(args.ts_col),
        score_col=(str(args.score_col) if args.score_col else None),
        confidence_col=(str(args.confidence_col) if args.confidence_col else None),
        reason_code=str(args.reason_code),
        model_version=str(args.model_version),
        threshold_snapshot_id=(str(args.snapshot_id) if args.snapshot_id else None),
        on_state=str(args.on_state),
        off_state=str(args.off_state),
    )

    ts_now = pd.Timestamp.utcnow().isoformat()
    store.insert_data_quality(
        ts=ts_now,
        symbol=str(args.symbol),
        missing_bars=int(args.missing_bars),
        stale_age_sec=float(args.stale_age_sec),
        source_name=str(args.source_name),
        fallback_used=bool(args.fallback_used),
    )

    latest = store.get_latest_state(str(args.symbol))
    print(
        "INFO ingested",
        inserted,
        "rows symbol=",
        args.symbol,
        "latest_ts=",
        None if latest is None else latest.get("ts"),
        "db=",
        store.db_path,
    )


if __name__ == "__main__":
    main()
