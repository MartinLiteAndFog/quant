"""Dedicated, paper-only entrypoint for a separately frozen v2 protocol."""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from quant.brain_forward.evidence import ForwardProtocol
from quant.brain_forward.runtime import FrozenUtilityMemory, completed_paper_trades
from quant.brain_forward.service import SOURCE, fetch_closed_binance_bars
from quant.brain_forward.store import (
    register_protocol,
    upsert_decisions,
    upsert_minute_bars,
    upsert_trades,
)
from quant.brain_forward.variant_store import upsert_variant_events, upsert_variant_trades
from quant.brain_forward.variants import evaluate_paper_variants


PROTOCOL = Path(__file__).with_name("forward_protocol_v2_20260824.json")
log = logging.getLogger("quant.brain_forward")


def run_once(
    symbol: str,
    model: FrozenUtilityMemory | None = None,
    protocol: ForwardProtocol | None = None,
) -> dict[str, int]:
    active_model = model or FrozenUtilityMemory()
    active_protocol = protocol or ForwardProtocol.load(PROTOCOL)
    active_protocol.assert_runtime(
        symbol=symbol,
        source=SOURCE,
        artifact_sha256=active_model.artifact_sha256,
    )
    bars = fetch_closed_binance_bars(symbol)
    if len(bars) < 1500:
        raise RuntimeError(f"need at least 1500 closed one-minute bars, received {len(bars)}")

    baseline_decisions, baseline_trades = completed_paper_trades(bars, active_model)
    evaluation = evaluate_paper_variants(bars, active_model)
    accepts = active_protocol.accepts_observation
    baseline_decisions = [item for item in baseline_decisions if accepts(item.event_ts)]
    baseline_trades = [item for item in baseline_trades if accepts(item["event_ts"])]
    variant_events = [item for item in evaluation.events if accepts(item["event_ts"])]
    variant_trades = [item for item in evaluation.trades if accepts(item["event_ts"])]

    register_protocol(active_protocol)
    result = {
        "bars": upsert_minute_bars(symbol, SOURCE, bars),
        "decisions": upsert_decisions(
            symbol,
            SOURCE,
            baseline_decisions,
            artifact_sha256=active_model.artifact_sha256,
            protocol=active_protocol,
        ),
        "trades": upsert_trades(
            symbol,
            SOURCE,
            baseline_trades,
            artifact_sha256=active_model.artifact_sha256,
            protocol=active_protocol,
        ),
        "variant_events": upsert_variant_events(
            variant_events,
            artifact_sha256=active_model.artifact_sha256,
            protocol=active_protocol,
        ),
        "variant_trades": upsert_variant_trades(
            variant_trades,
            artifact_sha256=active_model.artifact_sha256,
            protocol=active_protocol,
        ),
    }
    for event in variant_events:
        log.info(
            "paper variant candidate variant=%s event_ts=%s status=%s reason=%s trigger_ts=%s entry_price=%s",
            event["variant_id"],
            event["event_ts"],
            event["status"],
            event["reason"],
            event.get("trigger_ts"),
            event.get("entry_price"),
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Paper-only frozen Brain forward observer v2")
    parser.add_argument("--once", action="store_true", help="process recent closed bars once and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    if os.getenv("LIVE_TRADING_ENABLED", "0") not in ("", "0", "false", "False"):
        raise RuntimeError("brain forward observer refuses to run when LIVE_TRADING_ENABLED is enabled")
    protocol = ForwardProtocol.load(PROTOCOL)
    symbol = os.getenv("BRAIN_FORWARD_SYMBOL", protocol.symbol)
    if symbol != protocol.symbol:
        raise RuntimeError("BRAIN_FORWARD_SYMBOL differs from frozen v2 protocol")
    try:
        info = run_once(symbol, protocol=protocol)
        log.info("paper observer v2 refreshed symbol=%s %s", symbol, info)
        return 0
    except Exception:
        log.exception("paper observer v2 refresh failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
