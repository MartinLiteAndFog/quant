# src/quant/backtest/follow_runner.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# IO helpers
# -----------------------------
def _read_parquet_any(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "ts"
    df = df.sort_index()
    return df


def _read_signals_jsonl(path: Path) -> pd.DataFrame:
    """
    Reads your TradingView webhook JSONL (mixed possible formats),
    returns a dataframe with columns: ts (UTC), signal in {-1,0,+1}.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # try typical payload shapes:
            # { "ts": "...", "signal": 1 } or { "timestamp": "...", "action": "buy" } etc
            ts = obj.get("ts") or obj.get("timestamp") or obj.get("time") or obj.get("t")
            if ts is None:
                # some logs wrap payload
                payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else None
                if payload:
                    ts = payload.get("ts") or payload.get("timestamp") or payload.get("time")

            if ts is None:
                continue

            try:
                ts = pd.to_datetime(ts, utc=True)
            except Exception:
                continue

            sig = obj.get("signal")
            if sig is None:
                payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else None
                if payload:
                    sig = payload.get("signal")

            if sig is None:
                # map action/position to signal
                action = (obj.get("action") or "").lower()
                if not action and isinstance(obj.get("payload"), dict):
                    action = (obj["payload"].get("action") or "").lower()
                if action in ("buy", "long"):
                    sig = 1
                elif action in ("sell", "short"):
                    sig = -1
                else:
                    sig = 0

            try:
                sig = int(np.sign(float(sig)))
            except Exception:
                sig = 0

            rows.append((ts, sig))

    if not rows:
        return pd.DataFrame(columns=["ts", "signal"]).set_index("ts")

    df = pd.DataFrame(rows, columns=["ts", "signal"]).drop_duplicates(subset=["ts"]).sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int).clip(-1, 1)
    return df


def _align_signals_to_bricks(bricks: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
    """
    Forward-fill last known signal onto every brick timestamp.
    Assumes signals are sparse and should hold until changed.
    """
    if signals is None or len(signals) == 0:
        return pd.Series(0, index=bricks.index, dtype=int)

    sig = signals["signal"].copy()
    sig = sig[~sig.index.duplicated(keep="last")].sort_index()

    # merge_asof onto bricks
    tmp = pd.merge_asof(
        bricks.reset_index().sort_values("ts"),
        sig.reset_index().sort_values("ts"),
        on="ts",
        direction="backward",
    )
    out = tmp["signal"].fillna(0).astype(int).clip(-1, 1).to_numpy()
    return pd.Series(out, index=bricks.index, dtype=int)


# -----------------------------
# Backtest core (exit on opposite)
# -----------------------------
@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: int
    entry_px: float
    exit_px: float
    exit_event: str
    pnl_pct: float


@dataclass
class Event:
    ts: pd.Timestamp
    event: str
    side: int
    price: float
    pnl_pct: float
    note: str
    seq: int


def _pnl_pct(side: int, entry_px: float, exit_px: float) -> float:
    if entry_px <= 0 or exit_px <= 0:
        return 0.0
    raw = (exit_px / entry_px) - 1.0
    return float(side * raw)


def run_follow_opposite(
    bricks: pd.DataFrame,
    signal: pd.Series,
    fee_bps: float,
) -> Dict[str, pd.DataFrame]:
    """
    Strategy:
      - Enter when signal becomes +1/-1 while flat.
      - Exit when signal becomes 0 or flips sign (opposite).
      - If signal flips sign, we exit then immediately enter opposite (same ts/px).
    Fees:
      - fee_bps is PER SIDE (consistent with your observed 5 bps -> 10 bps RT shift).
      - roundtrip fee = 2 * fee_bps/10000.
      - flip on same bar pays exit fee + entry fee (2 sides).
    """
    fee_side = float(fee_bps) / 10000.0

    close = bricks["close"].to_numpy(dtype=float)
    ts = bricks.index

    pos = 0
    entry_px = np.nan
    entry_ts = None

    trades: List[Trade] = []
    events: List[Event] = []
    equity = 1.0
    equity_rows = []

    seq = 0

    def _log_event(t, ev, side, px, pnl, note):
        nonlocal seq
        events.append(Event(ts=t, event=ev, side=int(side), price=float(px), pnl_pct=float(pnl), note=str(note), seq=seq))
        seq += 1

    def _close_position(exit_t, exit_px, reason: str):
        nonlocal pos, entry_px, entry_ts, equity
        if pos == 0:
            return
        gross = _pnl_pct(pos, float(entry_px), float(exit_px))
        net = gross - 2.0 * fee_side  # entry + exit fee
        trades.append(
            Trade(
                entry_ts=entry_ts,
                exit_ts=exit_t,
                side=int(pos),
                entry_px=float(entry_px),
                exit_px=float(exit_px),
                exit_event=reason,
                pnl_pct=float(net),
            )
        )
        equity *= (1.0 + net)
        _log_event(exit_t, reason, pos, exit_px, net, f"EXIT {reason} (gross={gross:.6f}, fee_rt={2*fee_side:.6f})")
        pos = 0
        entry_px = np.nan
        entry_ts = None

    def _open_position(open_t, open_px, side: int):
        nonlocal pos, entry_px, entry_ts, equity
        if side == 0:
            return
        pos = int(np.sign(side))
        entry_px = float(open_px)
        entry_ts = open_t
        # pay entry fee immediately by reducing equity notionally? (equivalent to subtracting in trade net)
        # We already subtract in trade net, so only log.
        _log_event(open_t, "enter", pos, open_px, 0.0, f"ENTER side={pos} (fee_side={fee_side:.6f})")

    for i in range(len(bricks)):
        t = ts[i]
        px = close[i]
        sig = int(signal.iat[i])

        # record equity point each brick (lightweight)
        equity_rows.append((t, equity, pos, sig))

        if pos == 0:
            if sig != 0:
                _open_position(t, px, sig)
            continue

        # pos != 0
        if sig == 0:
            _close_position(t, px, "signal_exit_to_flat")
            continue

        if sig == -pos:
            # close and flip to new side (same brick)
            _close_position(t, px, "signal_flip_exit")
            _open_position(t, px, sig)
            continue

        # else: sig == pos -> hold

    # close any open position at end
    if pos != 0:
        _close_position(ts[-1], close[-1], "eod_exit")

    trades_df = pd.DataFrame([asdict(x) for x in trades])
    if len(trades_df) > 0:
        trades_df["entry_ts"] = pd.to_datetime(trades_df["entry_ts"], utc=True)
        trades_df["exit_ts"] = pd.to_datetime(trades_df["exit_ts"], utc=True)

    events_df = pd.DataFrame([asdict(x) for x in events])
    if len(events_df) > 0:
        events_df["ts"] = pd.to_datetime(events_df["ts"], utc=True)

    equity_df = pd.DataFrame(equity_rows, columns=["ts", "equity", "pos", "signal"])
    equity_df["ts"] = pd.to_datetime(equity_df["ts"], utc=True)
    equity_df = equity_df.set_index("ts")

    return {"trades": trades_df, "events": events_df, "equity": equity_df}


# -----------------------------
# CLI
# -----------------------------
def _max_drawdown(equity: pd.Series) -> float:
    x = equity.to_numpy(dtype=float)
    if len(x) == 0:
        return 0.0
    peak = np.maximum.accumulate(x)
    dd = (x / peak) - 1.0
    return float(dd.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--signals-jsonl", required=True)
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    sig_path = Path(args.signals_jsonl)

    bricks = _read_parquet_any(parquet_path)
    need = {"open", "high", "low", "close"}
    miss = [c for c in need if c not in bricks.columns]
    if miss:
        raise ValueError(f"Missing columns {miss} in {parquet_path}. cols={list(bricks.columns)[:30]}")

    sig_df = _read_signals_jsonl(sig_path)
    sig_series = _align_signals_to_bricks(bricks, sig_df)

    out_dir = Path("data/runs") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_follow_opposite(bricks=bricks, signal=sig_series, fee_bps=float(args.fee_bps))

    trades_df = res["trades"]
    events_df = res["events"]
    equity_df = res["equity"]

    # stats
    total_return_pct = (equity_df["equity"].iat[-1] - 1.0) * 100.0 if len(equity_df) else 0.0
    mdd_pct = _max_drawdown(equity_df["equity"]) * 100.0 if len(equity_df) else 0.0

    stats = {
        "start": str(bricks.index.min()),
        "end": str(bricks.index.max()),
        "rows": int(len(bricks)),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(mdd_pct),
        "fee_bps": float(args.fee_bps),
        "signals_jsonl": str(sig_path),
        "parquet": str(parquet_path),
        "trades": int(len(trades_df)),
        "strategy": "follow_exit_on_opposite_signal",
    }

    # write
    if len(trades_df):
        trades_df.to_parquet(out_dir / "trades.parquet", index=False)
    else:
        pd.DataFrame(columns=["entry_ts", "exit_ts", "side", "entry_px", "exit_px", "exit_event", "pnl_pct"]).to_parquet(
            out_dir / "trades.parquet", index=False
        )

    if len(events_df):
        events_df.to_parquet(out_dir / "events.parquet", index=False)
    else:
        pd.DataFrame(columns=["ts", "event", "side", "price", "pnl_pct", "note", "seq"]).to_parquet(
            out_dir / "events.parquet", index=False
        )

    equity_df.to_parquet(out_dir / "equity.parquet")

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"INFO stats {stats}")
    print(f"INFO wrote {out_dir}")


if __name__ == "__main__":
    main()
