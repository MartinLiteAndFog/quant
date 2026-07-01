from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from fastapi import Request

import quant.execution.webhook_server as ws
from quant.execution.webhook_server import tv_execute_webhook


def _json_request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/webhook/tv-execute",
        "raw_path": b"/webhook/tv-execute",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope, receive)


class TvExecuteWebhookTests(unittest.TestCase):
    def test_executor_exception_returns_structured_error(self) -> None:
        request = _json_request({"action": "entry", "side": "buy", "symbol": "SOL-USDT"})

        async def _run():
            from quant.execution.tv_signal_executor import TVSignal

            class _Cfg:
                symbol = "SOL-USDT"

                @classmethod
                def from_env(cls):
                    return cls()

            def _raise_execute(*args, **kwargs):
                raise RuntimeError("KuCoin API error: insufficient margin")

            with patch("quant.execution.webhook_server._auth_required", return_value=False), \
                 patch("quant.execution.tv_signal_executor._ready.is_set", return_value=True), \
                 patch("quant.execution.tv_signal_executor.parse_tv_signal", return_value=TVSignal(action="entry", side="buy", symbol="SOL-USDT")), \
                 patch("quant.execution.tv_signal_executor.execute_tv_signal", side_effect=_raise_execute), \
                 patch("quant.execution.tv_signal_executor.TVExecConfig", _Cfg), \
                 patch("asyncio.get_event_loop") as loop_mock:
                class _Loop:
                    async def run_in_executor(self, executor, fn, *args):
                        return fn(*args)

                loop_mock.return_value = _Loop()
                return await tv_execute_webhook(request)

        result = asyncio.run(_run())
        self.assertFalse(result["ok"])
        self.assertEqual(result["action"], "entry")
        self.assertEqual(result["symbol"], "SOL-USDT")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertIn("insufficient margin", result["reason"])

    def test_pre_execution_exception_returns_structured_error(self) -> None:
        request = _json_request({"action": "entry", "side": "sell", "symbol": "SOL-USDT"})

        async def _run():
            with patch("quant.execution.webhook_server._auth_required", return_value=False), \
                 patch("quant.execution.tv_signal_executor._ready.is_set", return_value=True), \
                 patch("quant.execution.tv_signal_executor.TVExecConfig.from_env", side_effect=RuntimeError("config exploded")):
                return await tv_execute_webhook(request)

        result = asyncio.run(_run())
        self.assertFalse(result["ok"])
        self.assertEqual(result["action"], "entry")
        self.assertEqual(result["symbol"], "SOL-USDT")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertIn("config exploded", result["reason"])


if __name__ == "__main__":
    unittest.main()
