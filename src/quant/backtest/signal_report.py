# src/quant/backtest/signal_report.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _read_equity(run_dir: Path) -> pd.DataFrame:
    p_parq = run_dir / "equity.parquet"
    p_csv = run_dir / "equity.csv"
    if p_parq.exists():
        df = pd.read_parquet(p_parq)
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"Missing equity.(parquet|csv) in {run_dir}")

    df = df.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")


def build_signal_report(equity: pd.DataFrame) -> Dict:
    if "pos" not in equity.columns:
        raise ValueError("equity curve must contain column 'pos'")

    pos = equity["pos"].fillna(0).astype(int).clip(-1, 1)

    n = int(len(pos))
    if n == 0:
        return {"rows": 0}

    # counts
    counts = pos.value_counts().to_dict()
    frac = {str(k): float(v) / n for k, v in counts.items()}

    # regime lengths (in bricks/rows)
    changes = (pos != pos.shift(1)).fillna(True)
    regime_id = changes.cumsum()
    regime_len = pos.groupby(regime_id).size()

    # number of switches (pos changes excluding first)
    switches = int((pos.diff().fillna(0) != 0).sum())

    report = {
        "rows": n,
        "pos_counts": {str(k): int(v) for k, v in counts.items()},
        "pos_fractions": {str(k): float(v) for k, v in frac.items()},
        "pos_switches": switches,
        "regime_len": {
            "count": int(regime_len.shape[0]),
            "mean": float(regime_len.mean()),
            "median": float(regime_len.median()),
            "p90": float(regime_len.quantile(0.90)),
            "max": int(regime_len.max()),
            "min": int(regime_len.min()),
        },
    }

    # If timestamps exist, add rough time stats (beware duplicate ts in renko)
    if "ts" in equity.columns:
        ts = equity["ts"]
        report["ts"] = {
            "start": str(ts.iloc[0]),
            "end": str(ts.iloc[-1]),
        }

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create signal/position quality report for a run folder")
    p.add_argument("--runs-dir", type=str, default="data/runs", help="Base runs directory (default: data/runs)")
    p.add_argument("--run", type=str, required=True, help="Run folder name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.runs_dir) / args.run

    eq = _read_equity(run_dir)
    report = build_signal_report(eq)

    out = run_dir / "signal_report.json"
    _write_json(out, report)
    print(f"Wrote: {out}")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
