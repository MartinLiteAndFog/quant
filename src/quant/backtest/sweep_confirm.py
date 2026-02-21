# src/quant/backtest/sweep_confirm.py
import itertools
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone

RUNS_DIR = Path("data/runs")


def run_cmd(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_stats(run_dir: Path) -> dict:
    """
    stats.json schema:
      {
        "meta": {...},
        "stats": {...}   # <- KPIs live here
      }
    """
    p = run_dir / "stats.json"
    if not p.exists():
        return {}
    obj = json.loads(p.read_text())
    # return the full object so we can access meta+stats if needed
    return obj if isinstance(obj, dict) else {}


def main():
    symbol = os.environ.get("SYMBOL", "SOL-USDT")
    tf = os.environ.get("TF", "1m")
    box = os.environ.get("BOX", "0.1")
    tp = os.environ.get("TP", "10")
    sl = os.environ.get("SL", "10")
    fee = os.environ.get("FEE", "4.0")

    parquet_path = os.environ.get(
        "PARQUET",
        "data/raw/exchange=kucoin/symbol=SOL-USDT/timeframe=1m/SOL-USDT_1m_20260207T102718Z.parquet",
    )
    signals_jsonl = os.environ.get(
        "SIGNALS_JSONL",
        f"data/signals/{symbol}/20260207.jsonl",
    )

    confirm_pct_grid = [0.0, 0.0005, 0.001, 0.002]
    timeout_grid = [10, 20, 40]
    fallback_grid = ["flat", "enter"]
    mode_grid = ["pct_from_exit"]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{symbol}_tf{tf}_box{box}_tp{tp}_sl{sl}_fee{fee}_sweep_{ts}"

    combos = list(itertools.product(confirm_pct_grid, timeout_grid, fallback_grid, mode_grid))
    print(f"Grid size: {len(combos)} runs. prefix={prefix}")

    for confirm_pct, timeout_bars, fallback, mode in combos:
        run_tag = f"cp{confirm_pct}_to{timeout_bars}_fb{fallback}_m{mode}".replace(".", "p")
        run_id = f"{prefix}__{run_tag}"

        cmd = [
            "quant-backtest",
            "--parquet", parquet_path,
            "--box", str(box),
            "--tp", str(tp),
            "--sl", str(sl),
            "--fee-bps", str(fee),
            "--signals-jsonl", str(signals_jsonl),
            "--run-id", run_id,
            "--flip-flat-then-enter",
            "--confirm-pct", str(confirm_pct),
            "--confirm-timeout-bars", str(timeout_bars),
            "--confirm-fallback", fallback,
            "--confirm-mode", mode,
        ]
        run_cmd(cmd)

    rows = []
    for d in RUNS_DIR.glob(f"{prefix}*"):
        if not d.is_dir():
            continue
        obj = load_stats(d)
        if not obj:
            continue

        st = obj.get("stats", {}) if isinstance(obj, dict) else {}
        if not isinstance(st, dict):
            st = {}

        rows.append(
            {
                "run_id": d.name,
                "total_return_pct": st.get("total_return_pct"),
                "max_drawdown_pct": st.get("max_drawdown_pct"),
                "turnover_sum": st.get("turnover_sum"),
                "rows": st.get("rows"),
                "start": st.get("start"),
                "end": st.get("end"),
                # optional: keep sweep params for easier filtering later
                "confirm_pct": obj.get("meta", {}).get("confirm_pct"),
                "confirm_timeout_bars": obj.get("meta", {}).get("confirm_timeout_bars"),
                "confirm_fallback": obj.get("meta", {}).get("confirm_fallback"),
                "confirm_mode": obj.get("meta", {}).get("confirm_mode"),
            }
        )

    def score(r):
        tr = r["total_return_pct"] or 0.0
        dd = abs(r["max_drawdown_pct"] or 0.0) or 1e-9
        return tr / dd

    rows.sort(key=score, reverse=True)

    out = Path("data") / "sweeps"
    out.mkdir(parents=True, exist_ok=True)
    out_json = out / f"{prefix}_summary.json"
    out_json.write_text(json.dumps(rows, indent=2))

    print(f"\nWrote: {out_json}")
    print("\nTop 10:")
    for r in rows[:10]:
        print(
            r["run_id"],
            r["total_return_pct"],
            r["max_drawdown_pct"],
            r["turnover_sum"],
            f"cp={r['confirm_pct']} to={r['confirm_timeout_bars']} fb={r['confirm_fallback']}",
        )


if __name__ == "__main__":
    main()
