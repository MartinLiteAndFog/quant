from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class NormalizedSignal:
    ts: pd.Timestamp          # UTC
    symbol: str               # e.g. "SOL-USDT"
    signal: int               # -1/0/+1
    action: str               # "buy"|"sell"|"close"
    raw: Dict[str, Any]


def _to_utc_ts(ts_value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(ts_value, utc=False, errors="raise")
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _normalize_symbol(symbol: str) -> str:
    s = (symbol or "").strip()
    if not s:
        raise ValueError("Missing symbol")
    return s.replace("/", "-").replace(":", "-")


def _action_to_signal(action: str) -> Tuple[str, int]:
    a = (action or "").strip().lower()
    if a in ("buy", "long", "entry_long", "open_long"):
        return "buy", 1
    if a in ("sell", "short", "entry_short", "open_short"):
        return "sell", -1
    if a in ("close", "flat", "exit", "close_all", "exit_long", "exit_short"):
        return "close", 0
    raise ValueError(f"Unknown action '{action}' (expected buy/sell/close)")


def normalize_payload(payload: Dict[str, Any]) -> NormalizedSignal:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")

    ts_raw = payload.get("ts") or payload.get("timestamp") or payload.get("time") or payload.get("server_ts")
    if not ts_raw:
        raise ValueError("Missing ts in payload (ts/timestamp/time/server_ts)")
    ts = _to_utc_ts(ts_raw)

    symbol = _normalize_symbol(payload.get("symbol") or payload.get("ticker") or payload.get("pair") or "")
    action = payload.get("action") or payload.get("side") or payload.get("order_action") or ""
    action_norm, sig = _action_to_signal(action)

    return NormalizedSignal(
        ts=ts,
        symbol=symbol,
        signal=sig,
        action=action_norm,
        raw=payload,
    )


def signals_day_path(symbol: str, ts_utc: pd.Timestamp) -> Path:
    safe_symbol = _normalize_symbol(symbol)
    day = ts_utc.strftime("%Y%m%d")
    base = Path("data/signals") / safe_symbol
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{day}.jsonl"


def append_signal_jsonl(sig: NormalizedSignal, *, include_server_ts: bool = True) -> Path:
    """
    Append one line to data/signals/<symbol>/<YYYYMMDD>.jsonl

    Always writes:
      server_ts (optional), ts, symbol, action, signal
    """
    path = signals_day_path(sig.symbol, sig.ts)

    out: Dict[str, Any] = {}
    if include_server_ts:
        out["server_ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    out["ts"] = sig.ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["symbol"] = sig.symbol
    out["action"] = sig.action
    out["signal"] = int(sig.signal)

    line = json.dumps(out, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    return path
