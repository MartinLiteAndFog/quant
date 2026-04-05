from __future__ import annotations

import unittest
from unittest.mock import patch

import quant.execution.tv_signal_executor as tv
from quant.execution.tv_signal_executor import TVCache, TVExecConfig, TVSignal


class TvSignalExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TVExecConfig(
            symbol="SOL-USDT",
            pos_pct=0.5,
            leverage=10.0,
            order_leverage=10.0,
            tp1_close_pct=0.5,
            dry_run=True,
            gate_mode="countertrend",
            cache_sec=10.0,
            cache_max_age_sec=60.0,
            emergency_sl_pct=0.023,
        )
        self._orig_broker = tv._broker
        tv._broker = object()

    def tearDown(self) -> None:
        tv._broker = self._orig_broker

    @staticmethod
    def _cache(*, position: float, current_side: str) -> TVCache:
        return TVCache(
            position=position,
            current_side=current_side,
            equity=1000.0,
            mid_price=100.0,
            bid=99.5,
            ask=100.5,
            contract_multiplier=1.0,
            qty=10,
            gate_on=0,
            gate_allows_entry=True,
            gate_source="test",
            updated_at=0.0,
        )

    def test_parse_tv_signal_allows_non_entry_side_labels(self) -> None:
        signal = tv.parse_tv_signal({"action": "sl", "side": "hold"}, default_symbol="SOL-USDT")
        self.assertEqual(signal.action, "sl")
        self.assertEqual(signal.side, "hold")
        self.assertEqual(signal.symbol, "SOL-USDT")

    def test_sl_buy_allowed_when_live_position_is_long(self) -> None:
        signal = TVSignal(action="sl", side="buy", symbol="SOL-USDT")
        cache = self._cache(position=3.0, current_side="long")

        with patch.object(tv, "_get_cache", return_value=cache):
            result = tv.execute_tv_signal(signal, self.config)

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "sl")
        self.assertEqual(result["reason"], "dry_run")
        self.assertEqual(result["qty"], 3)

    def test_sl_sell_allowed_when_live_position_is_long(self) -> None:
        signal = TVSignal(action="sl", side="sell", symbol="SOL-USDT")
        cache = self._cache(position=3.0, current_side="long")

        with patch.object(tv, "_get_cache", return_value=cache):
            result = tv.execute_tv_signal(signal, self.config)

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "sl")
        self.assertEqual(result["reason"], "dry_run")
        self.assertEqual(result["qty"], 3)

    def test_entry_cancels_existing_orders_before_submit(self) -> None:
        class _EntryBroker:
            def __init__(self) -> None:
                self.cancel_calls = []
                self.place_calls = []

            def cancel_all(self, symbol: str):
                self.cancel_calls.append(symbol)
                return {"ok": True}

            def place_market(self, symbol: str, side: str, qty: int, reduce_only: bool, client_id: str):
                self.place_calls.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "reduce_only": reduce_only,
                        "client_id": client_id,
                    }
                )
                return "oid-1"

        broker = _EntryBroker()
        signal = TVSignal(action="entry", side="sell", symbol="SOL-USDT")
        cache = self._cache(position=0.0, current_side="flat")

        with patch.object(tv, "_broker", broker), patch.object(tv, "_get_cache", return_value=cache):
            result = tv.execute_tv_signal(signal, self.config)

        self.assertTrue(result["ok"])
        self.assertEqual(broker.cancel_calls, ["SOL-USDT"])
        self.assertEqual(len(broker.place_calls), 1)
        self.assertFalse(broker.place_calls[0]["reduce_only"])

    def test_from_env_uses_tv_order_leverage_override(self) -> None:
        env = {
            "TV_EXEC_LEVERAGE": "8",
            "TV_EXEC_ORDER_LEVERAGE": "3",
        }
        with patch.dict("os.environ", env, clear=False):
            cfg = TVExecConfig.from_env()
        self.assertEqual(cfg.leverage, 8.0)
        self.assertEqual(cfg.order_leverage, 3.0)


if __name__ == "__main__":
    unittest.main()
