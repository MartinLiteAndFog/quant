from __future__ import annotations

import unittest

from quant.execution.cashflow_sync import (
    normalize_kraken_cashflows,
    normalize_kucoin_cashflows,
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


if __name__ == "__main__":
    unittest.main()
