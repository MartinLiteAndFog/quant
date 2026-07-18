from __future__ import annotations

import unittest
from unittest.mock import patch

from quant.execution.kucoin_futures import KucoinFuturesBroker, list_fills


class KucoinFuturesTests(unittest.TestCase):
    def test_strict_isolated_mode_never_falls_back_to_cross(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "KUCOIN_FUTURES_MARGIN_MODE": "isolated",
                "KUCOIN_FUTURES_STRICT_MARGIN_MODE": "1",
            },
            clear=False,
        ):
            broker = KucoinFuturesBroker(api_key="k", api_secret="s", passphrase="p")
            with patch.object(broker, "_position_margin_mode", return_value=None):
                self.assertEqual(broker._order_margin_mode_candidates("SOL-USDT"), ["ISOLATED"])

    def test_strict_isolated_mode_rejects_cross_position(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "KUCOIN_FUTURES_MARGIN_MODE": "isolated",
                "KUCOIN_FUTURES_STRICT_MARGIN_MODE": "1",
            },
            clear=False,
        ):
            broker = KucoinFuturesBroker(api_key="k", api_secret="s", passphrase="p")
            with patch.object(broker, "_position_margin_mode", return_value="CROSS"):
                with self.assertRaisesRegex(RuntimeError, "margin mode mismatch"):
                    broker._order_margin_mode_candidates("SOL-USDT")

    @patch("quant.execution.kucoin_futures._request")
    def test_list_fills_converts_seconds_to_milliseconds(self, mock_request) -> None:
        mock_request.return_value = {"items": []}
        list_fills(
            api_key="k",
            api_secret="s",
            passphrase="p",
            symbol="SOL-USDT",
            start_ts=1700000000,
            end_ts=1700000600,
            limit=25,
        )

        _, path = mock_request.call_args.args[:2]
        self.assertIn("from=1700000000000", path)
        self.assertIn("to=1700000600000", path)
        self.assertIn("pageSize=25", path)


if __name__ == "__main__":
    unittest.main()
