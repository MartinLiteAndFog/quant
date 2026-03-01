from __future__ import annotations

import unittest
from unittest.mock import patch

from quant.execution.kucoin_futures import list_fills


class KucoinFuturesTests(unittest.TestCase):
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
