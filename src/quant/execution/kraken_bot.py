from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from quant.execution.kraken_futures import KrakenFuturesClient
from quant.utils.log import get_logger

log = get_logger("quant.kraken_bot")


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _live_default(rel_path: str) -> str:
    if Path("/data").exists():
        return str(Path("/data/live/kraken") / rel_path)
    return str(Path("data/live/kraken") / rel_path)


def _now_iso() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_local_gate(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"ts": _now_iso(), "gate_on": 0, "gate_off": 1, "source": "local_missing"}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return {
            "ts": str(obj.get("ts") or _now_iso()),
            "gate_on": int(obj.get("gate_on", 0) or 0),
            "gate_off": int(obj.get("gate_off", 1) or 1),
            "source": "local_cache",
        }
    except Exception:
        return {"ts": _now_iso(), "gate_on": 0, "gate_off": 1, "source": "local_invalid"}


def _fetch_gate_http(url: str, timeout_s: int = 4) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "quant-kraken-bot/1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        out = r.read().decode("utf-8")
    obj = json.loads(out)
    return {
        "ts": str(obj.get("ts") or _now_iso()),
        "gate_on": int(obj.get("gate_on", 0) or 0),
        "gate_off": int(obj.get("gate_off", 1) or 1),
        "source": "api",
    }


def _publish_metrics(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _append_equity(path: Path, ts: str, equity_usd: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts_i = int(pd.Timestamp(ts).timestamp())
    snap = pd.DataFrame([{"ts": ts_i, "equity_usd": float(equity_usd)}])
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame(columns=["ts", "equity_usd"])
        df = pd.concat([old, snap], ignore_index=True)
    else:
        df = snap
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["equity_usd"] = pd.to_numeric(df["equity_usd"], errors="coerce")
    df = df.dropna(subset=["ts", "equity_usd"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    max_rows = int(max(200, int(os.getenv("KRAKEN_EQUITY_MAX_ROWS", "5000"))))
    if len(df) > max_rows:
        df = df.tail(max_rows)
    df.to_csv(path, index=False)


def _target_side_for_mode(strategy_mode: str) -> str:
    # Minimal live skeleton:
    # - countertrend mode defaults to short bias
    # - trend mode defaults to long bias
    # This can be replaced with strategy-specific signals later.
    return "short" if strategy_mode == "flip_countertrend" else "long"


def run_once(client: KrakenFuturesClient, gate_url: str, gate_cache_path: Path, metrics_path: Path, equity_path: Path) -> Dict[str, Any]:
    gate: Dict[str, Any]
    try:
        gate = _fetch_gate_http(gate_url)
        gate_cache_path.parent.mkdir(parents=True, exist_ok=True)
        gate_cache_path.write_text(json.dumps(gate, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        gate = _read_local_gate(gate_cache_path)
        gate["gate_error"] = str(e)

    gate_on = int(gate.get("gate_on", 0))
    strategy_mode = "flip_countertrend" if gate_on == 1 else "imbatrend_tp2"

    eq = client.get_account_equity()
    pos = client.get_position()
    mark = client.get_mark_price()

    desired_side = _target_side_for_mode(strategy_mode)
    dry_run = _truthy(os.getenv("KRAKEN_DRY_RUN", "1"))
    auto_trade = _truthy(os.getenv("KRAKEN_TRADING_ENABLED", "0"))
    target_size = float(os.getenv("KRAKEN_TARGET_SIZE", "0"))

    action = "hold"
    if auto_trade and not dry_run and target_size > 0:
        cur_side = str(pos.get("side") or "flat")
        cur_size = float(pos.get("size", 0) or 0)
        if cur_side == "flat" or cur_size <= 0:
            res = client.place_market("buy" if desired_side == "long" else "sell", size=target_size)
            action = f"enter_{desired_side}:{res.get('ok')}"
        elif cur_side != desired_side:
            res1 = client.close_position()
            res2 = client.place_market("buy" if desired_side == "long" else "sell", size=target_size)
            action = f"flip_{cur_side}_to_{desired_side}:{res1.get('ok')}:{res2.get('ok')}"

    ts = _now_iso()
    row = {
        "ts": ts,
        "equity_usd": float(eq.get("equity_usd", 0.0) or 0.0),
        "wallet_usd": float(eq.get("wallet_usd", 0.0) or 0.0),
        "upnl_usd": float(eq.get("upnl_usd", 0.0) or 0.0),
        "position_side": str(pos.get("side") or "flat"),
        "position_size": float(pos.get("size", 0.0) or 0.0),
        "mark_price": float(mark or 0.0),
        "leverage": float(os.getenv("KRAKEN_LEVERAGE", "10") or 10),
        "regime_state": "on" if gate_on == 1 else "off",
        "strategy_mode": strategy_mode,
        "gate_on": gate_on,
        "gate_off": int(gate.get("gate_off", 1)),
        "gate_ts": str(gate.get("ts")),
        "gate_source": str(gate.get("source", "unknown")),
        "action": action,
    }

    _publish_metrics(metrics_path, row)
    _append_equity(equity_path, ts=ts, equity_usd=float(row["equity_usd"]))

    # sanity logs
    gate_age_sec = max(0.0, (pd.Timestamp.now("UTC") - pd.Timestamp(row["gate_ts"])).total_seconds()) if row.get("gate_ts") else None
    log.info(
        "kraken-bot ts=%s gate_on=%s mode=%s gate_age_sec=%s equity=%s pos=%s/%s action=%s",
        ts,
        gate_on,
        strategy_mode,
        round(gate_age_sec, 2) if gate_age_sec is not None else None,
        row["equity_usd"],
        row["position_side"],
        row["position_size"],
        action,
    )
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kraken Futures SOL-USD execution bot")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll-sec", type=float, default=float(os.getenv("KRAKEN_POLL_SEC", "10")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = KrakenFuturesClient()
    gate_url = os.getenv("KRAKEN_GATE_URL", "http://127.0.0.1:8000/api/gate/solusd")
    gate_cache = Path(os.getenv("KRAKEN_GATE_CACHE_JSON", _live_default("gate_state.json")))
    metrics_path = Path(os.getenv("KRAKEN_METRICS_JSON", _live_default("metrics.json")))
    equity_path = Path(os.getenv("KRAKEN_EQUITY_CSV", _live_default("equity.csv")))

    while True:
        try:
            run_once(client, gate_url=gate_url, gate_cache_path=gate_cache, metrics_path=metrics_path, equity_path=equity_path)
        except Exception as e:
            log.warning("kraken-bot loop error: %s", e)
        if args.once:
            break
        time.sleep(max(1.0, float(args.poll_sec)))


if __name__ == "__main__":
    main()
