# scripts/sweep_grid.py
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _parse_grid(s: str) -> List[float]:
    if s is None or str(s).strip() == "":
        return []
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _fmt_num(x: float) -> str:
    if abs(x) < 1e-15:
        return "0"
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._=-]+", "_", s)
    return s[:240]


@dataclass
class RunResult:
    run_id: str
    rc: int
    stats: Dict


def _run_one(
    *,
    parquet: str,
    signals_jsonl: str,
    box: float,
    fee_bps: float,
    ttp: float,
    sl: float,
    lb: int,
    be_trig: float,
    be_off: float,
    sweep_tag: str,
    logs_dir: Path,
) -> RunResult:
    run_id = (
        f"SW_{sweep_tag}"
        f"_TTP{_fmt_num(ttp)}"
        f"_SL{_fmt_num(sl)}"
        f"_LB{int(lb)}"
        f"_BEt{_fmt_num(be_trig)}"
        f"_BEo{_fmt_num(be_off)}"
    )

    cmd = [
        "python",
        "-m",
        "quant.backtest.renko_runner",
        "--parquet",
        parquet,
        "--box",
        str(box),
        "--signals-jsonl",
        signals_jsonl,
        "--fee-bps",
        str(fee_bps),
        "--ttp-trail-pct",
        str(ttp),
        "--sl-cap-pct",
        str(sl),
        "--swing-lookback",
        str(int(lb)),
        "--be-trigger-pct",
        str(be_trig),
        "--be-offset-pct",
        str(be_off),
        "--run-id",
        run_id,
    ]

    log_path = logs_dir / f"{_safe_filename(run_id)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)

    stats_path = Path("data/runs") / run_id / "stats.json"
    stats: Dict = {}
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats = {}

    return RunResult(run_id=run_id, rc=int(rc), stats=stats)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--parquet", required=True)
    ap.add_argument("--signals-jsonl", required=True)
    ap.add_argument("--box", type=float, default=0.1)
    ap.add_argument("--fee-bps", type=float, default=4.0)

    ap.add_argument("--ttp-grid", required=True)
    ap.add_argument("--sl-grid", required=True)
    ap.add_argument("--lb-grid", required=True)

    ap.add_argument("--be-trigger-grid", default="0")
    ap.add_argument("--be-offset-grid", default="0")

    ap.add_argument("--top-n", type=int, default=30)

    args = ap.parse_args()

    ttp_grid = _parse_grid(args.ttp_grid)
    sl_grid = _parse_grid(args.sl_grid)
    lb_grid = [int(x) for x in _parse_grid(args.lb_grid)]

    be_trig_grid = _parse_grid(args.be_trigger_grid) or [0.0]
    be_off_grid = _parse_grid(args.be_offset_grid) or [0.0]

    sweep_tag = _now_tag()
    out_dir = Path("data/sweeps") / f"sweep_{sweep_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(ttp_grid, sl_grid, lb_grid, be_trig_grid, be_off_grid))
    total = len(combos)

    rows: List[Dict] = []
    for k, (ttp, sl, lb, be_trig, be_off) in enumerate(combos, start=1):
        rr = _run_one(
            parquet=str(args.parquet),
            signals_jsonl=str(args.signals_jsonl),
            box=float(args.box),
            fee_bps=float(args.fee_bps),
            ttp=float(ttp),
            sl=float(sl),
            lb=int(lb),
            be_trig=float(be_trig),
            be_off=float(be_off),
            sweep_tag=sweep_tag,
            logs_dir=logs_dir,
        )

        stats = rr.stats or {}
        ret = stats.get("total_return_pct", None)
        dd = stats.get("max_drawdown_pct", None)

        print(f"[{k}/{total}] {rr.run_id} rc={rr.rc} ret={ret} dd={dd}")

        rows.append(
            {
                "run_id": rr.run_id,
                "rc": rr.rc,
                "total_return_pct": ret,
                "max_drawdown_pct": dd,
                "ttp_trail_pct": float(ttp),
                "sl_cap_pct": float(sl),
                "swing_lookback": int(lb),
                "fee_bps": float(args.fee_bps),
                "be_trigger_pct": float(be_trig),
                "be_offset_pct": float(be_off),
                "stats_path": str(Path("data/runs") / rr.run_id / "stats.json"),
            }
        )

    df = pd.DataFrame(rows)

    results_csv = out_dir / "results.csv"
    results_pq = out_dir / "results.parquet"
    df.to_csv(results_csv, index=False)
    df.to_parquet(results_pq, index=False)

    df2 = df.dropna(subset=["total_return_pct"]).copy()
    df2["total_return_pct"] = pd.to_numeric(df2["total_return_pct"], errors="coerce")
    df2["max_drawdown_pct"] = pd.to_numeric(df2["max_drawdown_pct"], errors="coerce")
    df2 = df2.dropna(subset=["total_return_pct", "max_drawdown_pct"])

    topn = int(args.top_n)
    top = df2.sort_values(["total_return_pct", "max_drawdown_pct"], ascending=[False, False]).head(topn)
    top_csv = out_dir / f"top{topn}.csv"
    top.to_csv(top_csv, index=False)

    print(f"WROTE: {results_csv}")
    print(f"WROTE: {results_pq}")
    print(f"WROTE: {top_csv}")
    print(f"LOGS : {logs_dir}")


if __name__ == "__main__":
    main()
