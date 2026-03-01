from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant.execution.manual_orders import execute_manual_action


class _DummyBroker:
    def __init__(self, pos: float, bid: float = 82.0, ask: float = 82.1) -> None:
        self.pos = float(pos)
        self.bid = float(bid)
        self.ask = float(ask)
        self.place_calls = []
        self.wait_calls = []
        self.cancel_all_calls = 0

    def get_position(self, symbol: str) -> float:
        return self.pos

    def get_best_bid_ask(self, symbol: str):
        return (self.bid, self.ask)

    def place_marketable_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        self.place_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "limit_price": float(limit_price),
                "reduce_only": bool(reduce_only),
                "client_id": str(client_id),
            }
        )
        return "order-123"

    def wait_filled(self, symbol: str, order_id: str, timeout_s: int) -> bool:
        self.wait_calls.append((symbol, order_id, int(timeout_s)))
        return True

    def cancel_all(self, symbol: str) -> None:
        self.cancel_all_calls += 1


class ManualOrdersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        os.environ["MANUAL_ORDERS_LOG_JSONL"] = str(self.tmp_path / "manual_orders.jsonl")

    def tearDown(self) -> None:
        os.environ.pop("MANUAL_ORDERS_LOG_JSONL", None)
        self.tmp.cleanup()

    @patch("quant.execution.manual_orders.write_execution_state")
    @patch("quant.execution.manual_orders.record_expected")
    def test_cancel_short_places_reduce_only_buy_and_records_expected(self, mock_record_expected, mock_write_state) -> None:
        broker = _DummyBroker(pos=-7.0)
        out = execute_manual_action(
            broker=broker,
            symbol="SOL-USDT",
            action="cancel_short",
            qty=None,
            wait_sec=2.0,
            dry_run=False,
        )

        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("action"), "flatten_short")
        self.assertEqual(len(broker.place_calls), 1)
        call = broker.place_calls[0]
        self.assertEqual(call["side"], "buy")
        self.assertTrue(call["reduce_only"])
        self.assertEqual(int(call["qty"]), 7)
        self.assertEqual(len(broker.wait_calls), 1)

        mock_record_expected.assert_called_once()
        expected_trade = mock_record_expected.call_args[0][0]
        self.assertEqual(expected_trade.action, "exit_sl")
        self.assertEqual(expected_trade.side, "short")
        self.assertIn("manual_flatten_short", str(expected_trade.note))
        mock_write_state.assert_called_once()

    @patch("quant.execution.manual_orders.write_execution_state")
    @patch("quant.execution.manual_orders.record_expected")
    def test_cancel_short_dry_run_does_not_send_order(self, mock_record_expected, mock_write_state) -> None:
        broker = _DummyBroker(pos=-3.0)
        out = execute_manual_action(
            broker=broker,
            symbol="SOL-USDT",
            action="cancel_short",
            qty=None,
            wait_sec=2.0,
            dry_run=True,
        )

        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("action"), "flatten_short")
        self.assertEqual(len(broker.place_calls), 0)
        self.assertEqual(len(broker.wait_calls), 0)
        mock_record_expected.assert_called_once()
        mock_write_state.assert_not_called()

    @patch("quant.execution.manual_orders.write_execution_state")
    @patch("quant.execution.manual_orders.record_expected")
    def test_flatten_when_already_flat_is_noop(self, mock_record_expected, mock_write_state) -> None:
        broker = _DummyBroker(pos=0.0)
        out = execute_manual_action(
            broker=broker,
            symbol="SOL-USDT",
            action="flatten",
            qty=None,
            wait_sec=2.0,
            dry_run=False,
        )

        self.assertTrue(out.get("ok"))
        self.assertTrue(out.get("noop"))
        self.assertEqual(out.get("reason"), "already_flat")
        mock_record_expected.assert_not_called()
        mock_write_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
