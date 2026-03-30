from __future__ import annotations

import unittest

from quant.execution.kraken_futures import KrakenFuturesClient


class KrakenFuturesClientPriceTests(unittest.TestCase):
    def test_place_take_profit_market_rounds_stop_price_to_tick_size(self) -> None:
        client = KrakenFuturesClient()
        calls: list[tuple[str, str, dict, bool]] = []

        def fake_req(method: str, path: str, params=None, private: bool = False):
            payload = dict(params or {})
            calls.append((method, path, payload, private))
            if path == "/derivatives/api/v3/instruments":
                return {"instruments": [{"symbol": client.symbol, "tickSize": 0.01}]}
            return {"result": "success", "sendStatus": {"status": "placed", "order_id": "tp-1"}}

        client._req = fake_req  # type: ignore[method-assign]

        client.place_take_profit_market(
            side="sell",
            size=2.4,
            stop_price=85.592,
            symbol=client.symbol,
            reduce_only=True,
            cli_ord_id="tp-test",
        )

        self.assertEqual(calls[-1][1], "/derivatives/api/v3/sendorder")
        self.assertEqual(calls[-1][2]["stopPrice"], "85.59000000")

    def test_place_stop_market_rounds_stop_price_to_tick_size(self) -> None:
        client = KrakenFuturesClient()
        calls: list[tuple[str, str, dict, bool]] = []

        def fake_req(method: str, path: str, params=None, private: bool = False):
            payload = dict(params or {})
            calls.append((method, path, payload, private))
            if path == "/derivatives/api/v3/instruments":
                return {"instruments": [{"symbol": client.symbol, "tickSize": 0.01}]}
            return {"result": "success", "sendStatus": {"status": "placed", "order_id": "sl-1"}}

        client._req = fake_req  # type: ignore[method-assign]

        client.place_stop_market(
            side="sell",
            size=4.8,
            stop_price=80.537,
            symbol=client.symbol,
            reduce_only=True,
            cli_ord_id="sl-test",
        )

        self.assertEqual(calls[-1][1], "/derivatives/api/v3/sendorder")
        self.assertEqual(calls[-1][2]["stopPrice"], "80.54000000")


if __name__ == "__main__":
    unittest.main()
