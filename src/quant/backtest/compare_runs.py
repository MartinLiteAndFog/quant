# src/quant/backtest/compare_runs.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _read_stats(run_dir: Path) -> Dict:
    p = run_dir / "stats.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing stats.json in {run_dir}")
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return pd.read_json(p, typ="series").to_dict()


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


def _dedup_ts(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "none":
        return df
    if "ts" not in df.columns:
        return df
    if how not in ("last", "first"):
        raise ValueError("dedup mode must be one of: none, last, first")

    keep = how
    # stable order, then drop duplicates on ts
    out = df.sort_values("ts").drop_duplicates(subset=["ts"], keep=keep)
    return out


def _summarize(stats_obj: Dict, equity: pd.DataFrame) -> Dict:
    if "stats" in stats_obj and isinstance(stats_obj["stats"], dict):
        base = dict(stats_obj["stats"])
        meta = stats_obj.get("meta", {})
    else:
        base = dict(stats_obj)
        meta = {}

    out = dict(base)
    for k, v in meta.items():
        out[f"meta.{k}"] = v

    if len(equity) > 0 and "equity" in equity.columns:
        out["equity_start"] = float(equity["equity"].iloc[0])
        out["equity_end"] = float(equity["equity"].iloc[-1])
    if len(equity) > 0 and "pos" in equity.columns:
        out["pos_changes"] = int((equity["pos"].diff().fillna(0) != 0).sum())
    return out


def compare(a_dir: Path, b_dir: Path, dedup_ts: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a_stats = _read_stats(a_dir)
    b_stats = _read_stats(b_dir)

    a_eq = _dedup_ts(_read_equity(a_dir), dedup_ts)
    b_eq = _dedup_ts(_read_equity(b_dir), dedup_ts)

    a_sum = _summarize(a_stats, a_eq)
    b_sum = _summarize(b_stats, b_eq)

    keys = sorted(set(a_sum.keys()) | set(b_sum.keys()))
    rows = []
    for k in keys:
        av = a_sum.get(k, None)
        bv = b_sum.get(k, None)
        dv = None
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            dv = bv - av
        rows.append({"metric": k, "A": av, "B": bv, "B_minus_A": dv})

    table = pd.DataFrame(rows)

    if "ts" in a_eq.columns and "ts" in b_eq.columns and "equity" in a_eq.columns and "equity" in b_eq.columns:
        a_small = a_eq[["ts", "equity"]].rename(columns={"equity": "equity_A"})
        b_small = b_eq[["ts", "equity"]].rename(columns={"equity": "equity_B"})
        eq_join = pd.merge(a_small, b_small, on="ts", how="outer").sort_values("ts")
        eq_join["equity_A"] = eq_join["equity_A"].ffill()
        eq_join["equity_B"] = eq_join["equity_B"].ffill()
        eq_join["equity_ratio_B_over_A"] = eq_join["equity_B"] / eq_join["equity_A"]
    else:
        eq_join = pd.DataFrame()

    return table, eq_join


def _list_runs(runs_dir: Path, pattern: Optional[str]) -> List[str]:
    if not runs_dir.exists():
        return []
    dirs = [p.name for p in runs_dir.iterdir() if p.is_dir()]
    dirs = sorted(dirs)
    if pattern:
        pat = pattern.lower()
        dirs = [d for d in dirs if pat in d.lower()]
    return dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two backtest runs in data/runs/<run_id>")
    p.add_argument("--runs-dir", type=str, default="data/runs", help="Base runs directory (default: data/runs)")

    p.add_argument("--a", type=str, default=None, help="Run ID A (folder name)")
    p.add_argument("--b", type=str, default=None, help="Run ID B (folder name)")

    p.add_argument("--latest", type=int, default=None, help="Compare the latest N runs (use N=2).")
    p.add_argument("--pattern", type=str, default=None, help="Filter run folder names (e.g. sigjsonl or sigposdir).")

    p.add_argument(
        "--dedup-ts",
        type=str,
        default="none",
        choices=["none", "last", "first"],
        help="If equity has duplicate timestamps (Renko), deduplicate by keeping first/last per ts.",
    )

    p.add_argument("--out", type=str, default=None, help="Optional output path to write comparison CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)

    if args.latest:
        runs = _list_runs(runs_dir, args.pattern)
        if len(runs) < args.latest:
            raise SystemExit(f"Not enough runs in {runs_dir} (have {len(runs)}, need {args.latest})")
        pick = runs[-args.latest :]
        if len(pick) != 2:
            raise SystemExit("For now, use --latest 2 (compares last two runs).")
        a_id, b_id = pick[0], pick[1]
    else:
        if not args.a or not args.b:
            raise SystemExit("Provide --a and --b, or use --latest 2")
        a_id, b_id = args.a, args.b

    a_dir = runs_dir / a_id
    b_dir = runs_dir / b_id

    table, eq_join = compare(a_dir, b_dir, dedup_ts=args.dedup_ts)

    show_keys = [
        "rows",
        "total_return_pct",
        "max_drawdown_pct",
        "turnover_sum",
        "equity_start",
        "equity_end",
        "pos_changes",
        "start",
        "end",
        "meta.signals_jsonl",
        "meta.box",
        "meta.tp_bricks",
        "meta.sl_bricks",
        "meta.fee_bps",
        "meta.parquet_path",
    ]
    subset = table[table["metric"].isin(show_keys)].copy().sort_values("metric")

    print("\n=== RUN A ===")
    print(str(a_dir))
    print("=== RUN B ===")
    print(str(b_dir))
    print(f"\n=== METRICS (subset) [dedup-ts={args.dedup_ts}] ===")
    print(subset.to_string(index=False))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(outp, index=False)
        print(f"\nWrote: {outp}")

    if len(eq_join) > 0:
        print("\n=== EQUITY (tail 10 aligned by ts) ===")
        print(eq_join.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
