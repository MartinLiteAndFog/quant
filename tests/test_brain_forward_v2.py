from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from quant.brain_forward.evidence import ForwardProtocol
from quant.brain_forward import service_v2


class BrainForwardV2Tests(unittest.TestCase):
    def test_v2_protocol_pins_the_complete_v2_observer(self) -> None:
        protocol = ForwardProtocol.load(service_v2.PROTOCOL)
        self.assertEqual(protocol.schema_version, 2)
        self.assertFalse(protocol.live_orders_permitted)
        protocol.assert_runtime(
            symbol=protocol.symbol,
            source=protocol.source,
            artifact_sha256=protocol.artifact_sha256,
        )

    def test_v2_entrypoint_passes_the_frozen_protocol_to_the_paper_runner(self) -> None:
        with patch.dict(os.environ, {"LIVE_TRADING_ENABLED": "0", "BRAIN_FORWARD_SYMBOL": "SOL-USDT"}, clear=False), \
             patch.object(service_v2, "run_once", return_value={
                 "bars": 1599, "decisions": 0, "trades": 0,
                 "variant_events": 0, "variant_trades": 0,
             }) as run:
            self.assertEqual(service_v2.main(["--once"]), 0)
        self.assertEqual(run.call_args.args[0], "SOL-USDT")
        self.assertEqual(run.call_args.kwargs["protocol"].protocol_id, "brain-forward-sol-5m-v2-20260824")

    def test_v2_entrypoint_refuses_live_environment(self) -> None:
        with patch.dict(os.environ, {"LIVE_TRADING_ENABLED": "1"}, clear=False), \
             self.assertRaisesRegex(RuntimeError, "refuses"):
            service_v2.main(["--once"])

    def test_v2_run_once_writes_baseline_and_all_variant_ledgers(self) -> None:
        bars = pd.DataFrame({"ts": range(1500)})
        decision = SimpleNamespace(event_ts=pd.Timestamp("2026-08-24T00:00:00Z"))
        baseline_trade = {"event_ts": decision.event_ts}
        variant_event = {
            "event_ts": decision.event_ts, "variant_id": "immediate",
            "status": "triggered", "reason": "next_minute_open",
            "trigger_ts": decision.event_ts + pd.Timedelta(minutes=1),
            "entry_price": 100.0,
        }
        variant_trade = {"event_ts": decision.event_ts, "variant_id": "immediate"}
        model = SimpleNamespace(artifact_sha256="a" * 64)
        protocol = SimpleNamespace(
            assert_runtime=lambda **_: None,
            accepts_observation=lambda _: True,
        )
        evaluation = SimpleNamespace(
            events=[variant_event], trades=[variant_trade]
        )
        with patch.object(service_v2, "fetch_closed_binance_bars", return_value=bars), \
             patch.object(service_v2, "completed_paper_trades", return_value=([decision], [baseline_trade])), \
             patch.object(service_v2, "evaluate_paper_variants", return_value=evaluation), \
             patch.object(service_v2, "register_protocol") as register, \
             patch.object(service_v2, "upsert_minute_bars", return_value=1500), \
             patch.object(service_v2, "upsert_decisions", return_value=1) as decisions, \
             patch.object(service_v2, "upsert_trades", return_value=1) as trades, \
             patch.object(service_v2, "upsert_variant_events", return_value=1) as events, \
             patch.object(service_v2, "upsert_variant_trades", return_value=1) as variant_trades:
            result = service_v2.run_once("SOL-USDT", model=model, protocol=protocol)

        self.assertEqual(result, {
            "bars": 1500, "decisions": 1, "trades": 1,
            "variant_events": 1, "variant_trades": 1,
        })
        register.assert_called_once_with(protocol)
        self.assertEqual(decisions.call_args.args[2], [decision])
        self.assertEqual(trades.call_args.args[2], [baseline_trade])
        self.assertEqual(events.call_args.args[0], [variant_event])
        self.assertEqual(variant_trades.call_args.args[0], [variant_trade])


if __name__ == "__main__":
    unittest.main()
