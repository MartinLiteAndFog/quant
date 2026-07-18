from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from quant.execution.railway_entrypoint import main


class RailwayEntrypointTests(unittest.TestCase):
    def test_bot_process_dispatches_to_bot_launcher(self) -> None:
        with (
            patch.dict(os.environ, {"RAILWAY_PROCESS": "bot"}, clear=False),
            patch("quant.execution.railway_bot.main") as bot_main,
        ):
            main()
        bot_main.assert_called_once_with()

    def test_unknown_process_fails_closed(self) -> None:
        with patch.dict(os.environ, {"RAILWAY_PROCESS": "unknown"}, clear=False):
            with self.assertRaisesRegex(ValueError, "web.*bot"):
                main()


if __name__ == "__main__":
    unittest.main()
