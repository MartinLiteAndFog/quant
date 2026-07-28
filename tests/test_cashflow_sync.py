from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from quant.execution.cashflow_sync import (
    normalize_kraken_closed_trades,
    normalize_kraken_cashflows,
    normalize_kucoin_cashflows,
    sync_once,
)


class KucoinCashflowNormalizationTests(unittest.TestCase):
    def test_keeps_only_completed_transfers_with_authoritative_signs(self) -> None:
        rows = [
            {
                "time": 1_000_000,
                "type": "TransferIn",
                "amount": 40,
                "fee": 0,
                "accountEquity": 55,
                "status": "Completed",
                "remark": "Transferred from Funding Account",
                "offset": 101,
                "currency": "USDT",
            },
            {
                "time": 2_000_000,
                "type": "TransferOut",
                "amount": 5,
                "fee": 0,
                "accountEquity": 50,
                "status": "Completed",
                "remark": "Transferred to Funding Account",
                "offset": 102,
                "currency": "USDT",
            },
            {
                "time": 3_000_000,
                "type": "RealisedPNL",
                "amount": 9,
                "status": "Completed",
                "offset": 103,
                "currency": "USDT",
            },
            {
                "time": 4_000_000,
                "type": "TransferIn",
                "amount": 20,
                "status": "Pending",
                "offset": 104,
                "currency": "USDT",
            },
        ]
        events = normalize_kucoin_cashflows(rows, account="quant")
        self.assertEqual(len(events), 2)
        self.assertEqual([event["reporting_amount"] for event in events], [40.0, -5.0])
        self.assertEqual([event["direction"] for event in events], ["in", "out"])
        self.assertEqual(events[0]["equity_after"], 55.0)


class KrakenCashflowNormalizationTests(unittest.TestCase):
    def test_keeps_successful_routes_and_flags_missing_fx(self) -> None:
        rows = [
            {
                "id": "a",
                "date": "2026-04-01T10:00:00Z",
                "amount": "10",
                "asset": "usd",
                "from": "spot",
                "to": "futures",
                "status": "success",
            },
            {
                "id": "b",
                "date": "2026-04-02T10:00:00Z",
                "amount": "7",
                "asset": "eur",
                "from": "futures",
                "to": "spot",
                "status": "success",
            },
            {
                "id": "c",
                "date": "2026-04-03T10:00:00Z",
                "amount": "3",
                "asset": "usd",
                "from": "spot",
                "to": "futures",
                "status": "pending",
            },
        ]
        events = normalize_kraken_cashflows(rows, account="main")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["reporting_amount"], 10.0)
        self.assertEqual(events[1]["amount"], -7.0)
        self.assertIsNone(events[1]["reporting_amount"])
        self.assertEqual(events[1]["flow_type"], "futures_to_spot")


class KrakenClosedTradeNormalizationTests(unittest.TestCase):
    @staticmethod
    def _row(
        uid: str,
        ts: int,
        change: str,
        old_position: str,
        new_position: str,
        *,
        entry_price: str = "100",
        execution_price: str = "105",
        reason: str = "trade",
    ) -> dict:
        return {
            "uid": uid,
            "timestamp": ts,
            "event": {
                "PositionUpdate": {
                    "tradeable": "PF_SOLUSD",
                    "oldPosition": old_position,
                    "newPosition": new_position,
                    "oldAverageEntryPrice": entry_price,
                    "newAverageEntryPrice": execution_price,
                    "fillTime": ts,
                    "executionPrice": execution_price,
                    "executionSize": str(
                        abs(float(new_position) - float(old_position))
                    ),
                    "positionChange": change,
                    "updateReason": reason,
                    "realizedPnL": "5",
                    "fee": "0.1",
                    "feeCurrency": "USD",
                    "tradeType": "userExecution",
                }
            },
        }

    def test_keeps_only_completed_close_and_reverse_events(self) -> None:
        rows = [
            self._row("open", 1_000, "open", "0", "4"),
            self._row("partial", 2_000, "decrease", "4", "2"),
            self._row("reverse", 3_000, "reverse", "2", "-3"),
            self._row(
                "funding",
                4_000,
                "noChange",
                "-3",
                "-3",
                reason="fundingRealisation",
            ),
            self._row(
                "close",
                5_000,
                "close",
                "-3",
                "0",
                entry_price="105",
                execution_price="99",
            ),
        ]

        trades = normalize_kraken_closed_trades(reversed(rows))

        self.assertEqual(
            [trade["trade_id"] for trade in trades],
            ["kraken-position:reverse", "kraken-position:close"],
        )
        self.assertEqual([trade["side"] for trade in trades], ["long", "short"])
        self.assertEqual([trade["qty"] for trade in trades], [2.0, 3.0])
        self.assertEqual(trades[0]["entry_ts"].timestamp(), 1.0)
        self.assertEqual(trades[1]["entry_ts"].timestamp(), 3.0)
        self.assertAlmostEqual(trades[0]["pnl_pct"], 5.0)
        self.assertAlmostEqual(
            trades[1]["pnl_pct"],
            100.0 * (1.0 - 99.0 / 105.0),
        )

    def test_ignores_other_markets_and_non_trade_updates(self) -> None:
        other_market = self._row("other", 1_000, "close", "1", "0")
        other_market["event"]["PositionUpdate"]["tradeable"] = "PF_XBTUSD"
        funding = self._row(
            "funding",
            2_000,
            "close",
            "1",
            "0",
            reason="fundingRealisation",
        )

        self.assertEqual(
            normalize_kraken_closed_trades([other_market, funding]),
            [],
        )


class KrakenSyncTests(unittest.TestCase):
    @patch("quant.execution.cashflow_sync.upsert_cashflow_sync_state")
    @patch("quant.execution.cashflow_sync.upsert_closed_trade")
    @patch("quant.execution.cashflow_sync.upsert_cashflow_event")
    @patch("quant.execution.cashflow_sync._fetch_kraken_closed_trades")
    @patch("quant.execution.cashflow_sync._fetch_kraken_cashflows")
    @patch("quant.execution.cashflow_sync.ensure_cashflow_schema")
    def test_persists_cashflows_and_completed_trades(
        self,
        ensure_schema,
        fetch_cashflows,
        fetch_trades,
        upsert_cashflow,
        upsert_trade,
        upsert_state,
    ) -> None:
        fetch_cashflows.return_value = [{"event_id": "flow"}]
        fetch_trades.return_value = [{"trade_id": "trade"}]

        count = sync_once(venue="kraken", account="main")

        self.assertEqual(count, 1)
        ensure_schema.assert_called_once()
        upsert_cashflow.assert_called_once_with({"event_id": "flow"})
        upsert_trade.assert_called_once_with({"trade_id": "trade"})
        self.assertIsNone(upsert_state.call_args.kwargs["last_error"])

    @patch("quant.execution.cashflow_sync.upsert_cashflow_sync_state")
    @patch("quant.execution.cashflow_sync.upsert_cashflow_event")
    @patch(
        "quant.execution.cashflow_sync._fetch_kraken_closed_trades",
        side_effect=RuntimeError("history unavailable"),
    )
    @patch("quant.execution.cashflow_sync._fetch_kraken_cashflows")
    @patch("quant.execution.cashflow_sync.ensure_cashflow_schema")
    def test_trade_history_failure_does_not_leave_ledger_pending(
        self,
        ensure_schema,
        fetch_cashflows,
        fetch_trades,
        upsert_cashflow,
        upsert_state,
    ) -> None:
        fetch_cashflows.return_value = [{"event_id": "flow"}]

        count = sync_once(venue="kraken", account="main")

        self.assertEqual(count, 1)
        upsert_cashflow.assert_called_once_with({"event_id": "flow"})
        self.assertIsNone(upsert_state.call_args.kwargs["last_error"])


class CashflowSyncStateSqlTests(unittest.TestCase):
    def test_null_error_parameter_has_explicit_postgres_type(self) -> None:
        from quant.execution.event_store import upsert_cashflow_sync_state

        captured: dict = {}

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params):
                captured["sql"] = sql
                captured["params"] = params

        class Connection:
            def cursor(self):
                return Cursor()

        @contextmanager
        def fake_conn():
            yield Connection()

        with patch(
            "quant.execution.event_store.ensure_cashflow_schema"
        ), patch(
            "quant.execution.event_store.get_conn",
            side_effect=fake_conn,
        ):
            upsert_cashflow_sync_state(
                venue="kraken",
                account="main",
                coverage_start=None,
                coverage_end=None,
                source="test",
                last_error=None,
            )

        self.assertIn(
            "case when %(last_error)s::text is null",
            captured["sql"],
        )
        self.assertIsNone(captured["params"]["last_error"])


if __name__ == "__main__":
    unittest.main()
