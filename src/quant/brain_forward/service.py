"""Railway entrypoint for a paper-only frozen Brain forward observer."""

from __future__ import annotations

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from quant.brain_forward.evidence import ForwardProtocol
from quant.brain_forward.runtime import FrozenUtilityMemory, completed_paper_trades, parse_binance_klines
from quant.brain_forward.store import register_protocol, upsert_decisions, upsert_minute_bars, upsert_trades


log = logging.getLogger("quant.brain_forward")
SOURCE = "binance_spot_klines"


def _request_klines(symbol: str, *, end_time: int | None = None) -> list[list[Any]]:
    params: dict[str, object] = {"symbol": symbol.replace("-", ""), "interval": "1m", "limit": 1000}
    if end_time is not None:
        params["endTime"] = end_time
    with urlopen(f"https://api.binance.com/api/v3/klines?{urlencode(params)}", timeout=15) as response:  # nosec B310: fixed HTTPS endpoint
        payload: Any = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Binance kline response was not a list")
    return payload


def fetch_closed_binance_bars(symbol: str, limit: int = 1600) -> pd.DataFrame:
    """Page Binance's 1,000-kline cap to retain the 1,440-minute volatility state."""

    newest = _request_klines(symbol)
    if not newest:
        return parse_binance_klines([])
    first_open = int(newest[0][0])
    older = _request_klines(symbol, end_time=first_open - 1) if limit > 1000 else []
    rows = (older + newest)[-int(limit):]
    return parse_binance_klines(rows)


def run_once(
    symbol: str,
    model: FrozenUtilityMemory | None = None,
    protocol: ForwardProtocol | None = None,
) -> dict[str, int]:
    active_model = model or FrozenUtilityMemory()
    active_protocol = protocol or ForwardProtocol.load()
    active_protocol.assert_runtime(symbol=symbol, source=SOURCE, artifact_sha256=active_model.artifact_sha256)
    bars = fetch_closed_binance_bars(symbol)
    if len(bars) < 1500:
        raise RuntimeError(f"need at least 1500 closed one-minute bars, received {len(bars)}")
    decisions, trades = completed_paper_trades(bars, active_model)
    decisions = [decision for decision in decisions if active_protocol.accepts_observation(decision.event_ts)]
    trades = [trade for trade in trades if active_protocol.accepts_observation(trade["event_ts"])]
    register_protocol(active_protocol)
    return {
        "bars": upsert_minute_bars(symbol, SOURCE, bars),
        "decisions": upsert_decisions(symbol, SOURCE, decisions, artifact_sha256=active_model.artifact_sha256, protocol=active_protocol),
        "trades": upsert_trades(symbol, SOURCE, trades, artifact_sha256=active_model.artifact_sha256, protocol=active_protocol),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Paper-only frozen Brain forward observer")
    parser.add_argument("--once", action="store_true", help="process recent closed bars once and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    symbol = os.getenv("BRAIN_FORWARD_SYMBOL", "SOL-USDT")
    poll_seconds = float(os.getenv("BRAIN_FORWARD_POLL_SEC", "60"))
    if os.getenv("LIVE_TRADING_ENABLED", "0") not in ("", "0", "false", "False"):
        raise RuntimeError("brain forward observer refuses to run when LIVE_TRADING_ENABLED is enabled")
    if args.once:
        try:
            info = run_once(symbol)
            log.info("paper observer refreshed symbol=%s %s", symbol, info)
            return 0
        except Exception:
            log.exception("paper observer refresh failed")
            return 1
    while True:
        try:
            info = run_once(symbol)
            log.info("paper observer refreshed symbol=%s %s", symbol, info)
        except Exception:
            log.exception("paper observer refresh failed")
        time.sleep(max(10.0, poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
