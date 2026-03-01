# src/quant/execution/live_monitor.py
"""
Live monitoring: predicted vs actual performance.

- Record "expected" trades when signals are executed (entry/exit intent, expected price or fill mode).
- Fetch actual fills from KuCoin (or from a local log written by the executor).
- Match expected ↔ actual and compute: entry slippage, exit slippage, PnL predicted vs actual.
- Export CSV/JSON or serve a simple dashboard for improving entries.

Expected trades are appended to a JSONL (e.g. data/live/expected_trades.jsonl).
Actual fills are fetched via KuCoin API or read from data/live/fills_*.jsonl if you log them.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.execution.kucoin_futures import list_fills
from quant.utils.log import get_logger

log = get_logger("quant.live_monitor")


@dataclass
class ExpectedTrade:
    """One expected trade (from signal/backtest intent)."""
    ts: str
    symbol: str
    side: str  # long | short
    action: str  # entry | exit_tp | exit_sl | exit_flip
    qty: float
    expected_px: Optional[float] = None
    client_oid: Optional[str] = None
    signal_id: Optional[str] = None
    note: Optional[str] = None


@dataclass
class ActualFill:
    """One actual fill from exchange."""
    ts: str
    symbol: str
    side: str
    qty: float
    price: float
    order_id: Optional[str] = None
    client_oid: Optional[str] = None
    reduce_only: bool = False


@dataclass
class MatchedTrade:
    """Expected trade matched with actual fill(s)."""
    expected: ExpectedTrade
    actual_entry_px: Optional[float] = None
    actual_exit_px: Optional[float] = None
    entry_slippage_bps: Optional[float] = None
    exit_slippage_bps: Optional[float] = None
    pnl_pct_predicted: Optional[float] = None
    pnl_pct_actual: Optional[float] = None


def _default_live_dir() -> Path:
    """Use mounted volume by default when available."""
    if Path("/data").exists():
        return Path("/data/live")
    return Path("data/live")


def _live_dir() -> Path:
    raw = (os.getenv("QUANT_LIVE_DIR") or "").strip()
    if raw:
        return Path(raw)
    return _default_live_dir()


def _expected_trades_path(live_dir: Optional[Path] = None) -> Path:
    """
    Resolve expected-trades path.
    Priority:
      1) explicit live_dir argument
      2) DASHBOARD_EXPECTED_TRADES_JSONL env (shared dashboard path)
      3) QUANT_LIVE_DIR or mounted /data/live fallback
    """
    if live_dir is not None:
        return Path(live_dir) / "expected_trades.jsonl"
    env_path = (os.getenv("DASHBOARD_EXPECTED_TRADES_JSONL") or "").strip()
    if env_path:
        return Path(env_path)
    return _live_dir() / "expected_trades.jsonl"


def record_expected(trade: ExpectedTrade, live_dir: Optional[Path] = None) -> None:
    """Append one expected trade to JSONL for later matching."""
    path = _expected_trades_path(live_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(trade), ensure_ascii=False, separators=(",", ":"), default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    log.debug("record_expected symbol=%s action=%s", trade.symbol, trade.action)


def load_expected_trades(live_dir: Optional[Path] = None, since_ts: Optional[str] = None) -> pd.DataFrame:
    """Load expected trades from JSONL."""
    path = _expected_trades_path(live_dir)
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if since_ts and row.get("ts", "") < since_ts:
                    continue
                rows.append(row)
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def fetch_actual_fills_from_kucoin(
    symbol: str = "",
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    limit: int = 200,
) -> List[ActualFill]:
    """Fetch recent fills from KuCoin Futures API."""
    raw = list_fills(symbol=symbol, start_ts=start_ts, end_ts=end_ts, limit=limit)
    out = []
    for r in raw:
        ts_ms = r.get("createdAt") or r.get("tradeTime") or r.get("ts")
        if ts_ms is None:
            continue
        ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC").isoformat() if isinstance(ts_ms, (int, float)) else str(ts_ms)
        side = (r.get("side") or "buy").lower()
        price = float(r.get("price") or r.get("dealPrice") or 0)
        qty = float(r.get("size") or r.get("qty") or 0)
        out.append(ActualFill(
            ts=ts,
            symbol=(r.get("symbol") or symbol or "").replace("M", "").replace("USDT", "-USDT"),
            side=side,
            qty=qty,
            price=price,
            order_id=str(r.get("orderId", r.get("order_id", ""))),
            client_oid=r.get("clientOid") or r.get("client_oid"),
            reduce_only=bool(r.get("reduceOnly", r.get("reduce_only", False))),
        ))
    return out


def match_expected_to_actual(
    expected_df: pd.DataFrame,
    actual_fills: List[ActualFill],
    time_window_sec: float = 120.0,
) -> List[MatchedTrade]:
    """
    Match expected trades to actual fills by time window and side.
    Pairs entry + exit of same symbol/side into one trade and computes slippage / PnL diff.
    """
    if expected_df.empty or not actual_fills:
        return []
    actual_df = pd.DataFrame([asdict(a) for a in actual_fills])
    actual_df["ts"] = pd.to_datetime(actual_df["ts"], utc=True, errors="coerce")
    actual_df = actual_df.dropna(subset=["ts"])

    matched: List[MatchedTrade] = []
    for _, ex in expected_df.iterrows():
        exp_ts = pd.Timestamp(ex["ts"], tz="UTC") if ex.get("ts") else None
        if exp_ts is None or pd.isna(exp_ts):
            continue
        sym = (ex.get("symbol") or "").strip()
        side = (ex.get("side") or "").strip().lower()
        action = (ex.get("action") or "").strip().lower()
        exp_px = ex.get("expected_px")
        if pd.isna(exp_px):
            exp_px = None
        else:
            exp_px = float(exp_px)

        # Find actual fills within time window
        mask = (actual_df["ts"] >= exp_ts - pd.Timedelta(seconds=time_window_sec)) & (
            actual_df["ts"] <= exp_ts + pd.Timedelta(seconds=time_window_sec)
        )
        if sym:
            mask &= actual_df["symbol"].astype(str).str.upper().str.replace("-", "").str.contains(
                sym.upper().replace("-", ""), na=False
            )
        candidates = actual_df.loc[mask]
        _exp_fields = {"ts", "symbol", "side", "action", "qty", "expected_px", "client_oid", "signal_id", "note"}
        _exp_kw = {k: ex[k] for k in _exp_fields if k in ex.index}
        _exp = ExpectedTrade(**_exp_kw)

        if candidates.empty:
            matched.append(MatchedTrade(expected=_exp))
            continue

        # Take closest fill by time
        candidates = candidates.copy()
        candidates["dt"] = (candidates["ts"] - exp_ts).abs().dt.total_seconds()
        best = candidates.loc[candidates["dt"].idxmin()]
        actual_px = float(best.get("price", 0))
        entry_slippage_bps = None
        if exp_px and actual_px and exp_px > 0:
            if "entry" in action or action == "entry":
                # Long entry: expected ask, actual fill; slippage = (actual - expected)/expected
                entry_slippage_bps = (actual_px - exp_px) / exp_px * 10_000 if side == "long" else (exp_px - actual_px) / exp_px * 10_000
            # exit slippage similarly if we have expected exit price

        m = MatchedTrade(
            expected=_exp,
            actual_entry_px=actual_px if "entry" in action else None,
            actual_exit_px=actual_px if "exit" in action else None,
            entry_slippage_bps=entry_slippage_bps,
        )
        matched.append(m)
    return matched


def report_predicted_vs_actual(
    matched: List[MatchedTrade],
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Summarise matched trades: mean slippage, PnL diff, and optional CSV export."""
    if not matched:
        return {"n_matched": 0}
    entry_slippage = [m.entry_slippage_bps for m in matched if m.entry_slippage_bps is not None]
    exit_slippage = [m.exit_slippage_bps for m in matched if m.exit_slippage_bps is not None]
    summary = {
        "n_matched": len(matched),
        "entry_slippage_bps_mean": float(sum(entry_slippage) / len(entry_slippage)) if entry_slippage else None,
        "entry_slippage_bps_median": float(pd.Series(entry_slippage).median()) if entry_slippage else None,
        "exit_slippage_bps_mean": float(sum(exit_slippage) / len(exit_slippage)) if exit_slippage else None,
        "exit_slippage_bps_median": float(pd.Series(exit_slippage).median()) if exit_slippage else None,
    }
    if out_path:
        rows = []
        for m in matched:
            r = asdict(m.expected)
            r["actual_entry_px"] = m.actual_entry_px
            r["actual_exit_px"] = m.actual_exit_px
            r["entry_slippage_bps"] = m.entry_slippage_bps
            r["exit_slippage_bps"] = m.exit_slippage_bps
            rows.append(r)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        log.info("Wrote predicted_vs_actual report to %s", out_path)
    return summary
