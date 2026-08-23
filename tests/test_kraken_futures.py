from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from quant.execution.kraken_futures import KrakenFuturesClient


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


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


class KrakenFuturesPositionHistoryTests(unittest.TestCase):
    @patch("quant.execution.kraken_futures.urllib.request.urlopen")
    def test_reads_and_paginates_authenticated_position_events(self, urlopen) -> None:
        urlopen.side_effect = [
            _FakeResponse(
                {
                    "elements": [{"executionUid": "one"}],
                    "continuationToken": "next-page",
                }
            ),
            _FakeResponse({"elements": [{"executionUid": "two"}]}),
        ]
        with patch.dict(
            "os.environ",
            {
                "KRAKEN_FUTURES_KEY": "read-key",
                "KRAKEN_FUTURES_SECRET": "c2VjcmV0",
            },
        ):
            rows = KrakenFuturesClient().get_position_events(
                symbol="PF_SOLUSD",
                since_ms=123000,
                limit=2,
            )

        self.assertEqual([row["executionUid"] for row in rows], ["one", "two"])
        first_request = urlopen.call_args_list[0].args[0]
        second_request = urlopen.call_args_list[1].args[0]
        self.assertIn("/api/history/v3/positions?", first_request.full_url)
        self.assertIn("tradeable=PF_SOLUSD", first_request.full_url)
        self.assertIn("since=123000", first_request.full_url)
        self.assertIn("continuation_token=next-page", second_request.full_url)
        self.assertTrue(first_request.get_header("Apikey"))
        self.assertTrue(first_request.get_header("Authent"))

    @patch("quant.execution.kraken_futures.urllib.request.urlopen")
    def test_position_events_can_include_funding_without_changing_default(self, urlopen) -> None:
        urlopen.return_value = _FakeResponse({"elements": []})
        with patch.dict(
            "os.environ",
            {
                "KRAKEN_FUTURES_KEY": "read-key",
                "KRAKEN_FUTURES_SECRET": "c2VjcmV0",
            },
        ):
            KrakenFuturesClient().get_position_events(
                symbol="PF_SOLUSD", include_funding=True
            )
        request = urlopen.call_args.args[0]
        self.assertIn("trades=true", request.full_url)
        self.assertIn("funding_realization=true", request.full_url)
        self.assertIn("settlement=true", request.full_url)

    @patch("quant.execution.kraken_futures.urllib.request.urlopen")
    def test_position_history_unwraps_current_event_envelope(self, urlopen) -> None:
        urlopen.return_value = _FakeResponse(
            {
                "elements": [
                    {
                        "timestamp": 123_000,
                        "uid": "history-row-1",
                        "event": {
                            "PositionUpdate": {
                                "executionUid": "execution-1",
                                "updateReason": "trade",
                                "positionChange": "open",
                                "executionPrice": "75.5",
                            }
                        },
                    }
                ]
            }
        )
        with patch.dict(
            "os.environ",
            {
                "KRAKEN_FUTURES_KEY": "read-key",
                "KRAKEN_FUTURES_SECRET": "c2VjcmV0",
            },
        ):
            rows = KrakenFuturesClient().get_position_events(limit=1)

        self.assertEqual(rows[0]["executionUid"], "execution-1")
        self.assertEqual(rows[0]["timestamp"], 123_000)
        self.assertEqual(rows[0]["historyUid"], "history-row-1")


if __name__ == "__main__":
    unittest.main()
