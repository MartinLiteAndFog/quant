from __future__ import annotations

import unittest
import asyncio
from unittest.mock import patch

from quant.execution import kraken_tv_executor as ktv
from quant.execution.kraken_tv_executor import (
    KrakenTVConfig,
    compute_target_size,
    execute_kraken_tv_signal,
    parse_kraken_tv_signal,
)


class DummyKrakenClient:
    def __init__(
        self,
        position_signed: float = 0.0,
        equity_usd: float = 100.0,
        mark_price: float = 100.0,
    ) -> None:
        self.position_signed = position_signed
        self.equity_usd = equity_usd
        self.mark_price = mark_price
        self.market_orders: list[dict] = []
        self.cancel_calls: list[str] = []

    def get_mark_price(self, symbol=None) -> float:
        return self.mark_price

    def get_account_equity(self) -> dict:
        return {
            "wallet_usd": self.equity_usd,
            "upnl_usd": 0.0,
            "equity_usd": self.equity_usd,
        }

    def get_position(self, symbol=None) -> dict:
        side = "long" if self.position_signed > 0 else ("short" if self.position_signed < 0 else "flat")
        return {
            "side": side,
            "size": abs(self.position_signed),
            "size_signed": self.position_signed,
            "entry_price": 90.0,
            "leverage": 10.0,
            "raw": {"symbol": symbol},
        }

    def place_market(self, side: str, size: float, symbol=None, reduce_only: bool = False, cli_ord_id=None) -> dict:
        self.market_orders.append(
            {
                "side": side,
                "size": size,
                "symbol": symbol,
                "reduce_only": reduce_only,
            }
        )
        signed_delta = size if side == "buy" else -size
        self.position_signed += signed_delta
        return {"ok": True, "order_id": f"order-{len(self.market_orders)}"}

    def cancel_all_reduce_only_orders(self, symbol=None) -> dict:
        self.cancel_calls.append(str(symbol))
        return {"ok": True, "cancelled": []}


def _config(**overrides) -> KrakenTVConfig:
    values = {
        "venue_symbol": "PF_SOLUSD",
        "display_symbol": "SOL-USDT",
        "pos_pct": 0.90,
        "leverage": 10.0,
        "tp1_frac": 0.50,
        "dry_run": True,
        "size_step": 0.1,
        "dedup_ttl_sec": 0.0,
        "cancel_reduce_only_on_flip": True,
        "verify_after_order": True,
        "refill_partial": False,
    }
    values.update(overrides)
    return KrakenTVConfig(**values)


def _signal(action: str = "flip", side: str = "sell", bar_index: int = 1):
    return parse_kraken_tv_signal(
        {
            "action": action,
            "side": side,
            "symbol": "SOL-USDT",
            "source": "test",
            "reason": "unit",
            "bar_time": 123,
            "bar_index": bar_index,
        }
    )


class KrakenTVExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        ktv._SEEN_FINGERPRINTS.clear()

    def test_compute_target_size_floors_to_kraken_step(self) -> None:
        self.assertEqual(compute_target_size(100.0, 83.0, 10.0, 0.90, 0.1), 10.8)

    def test_flip_uses_single_net_order_from_current_signed_to_target_signed(self) -> None:
        client = DummyKrakenClient(position_signed=9.0, equity_usd=100.0, mark_price=100.0)
        res = execute_kraken_tv_signal(_signal("flip", "sell"), _config(dry_run=False), client)

        self.assertTrue(res["ok"])
        self.assertEqual(res["target_size"], 9.0)
        self.assertEqual(res["desired_signed"], -9.0)
        self.assertEqual(res["order_side"], "sell")
        self.assertEqual(res["order_size"], 18.0)
        self.assertEqual(client.market_orders, [{"side": "sell", "size": 18.0, "symbol": "PF_SOLUSD", "reduce_only": False}])
        self.assertEqual(client.position_signed, -9.0)
        self.assertEqual(client.cancel_calls, ["PF_SOLUSD"])

    def test_flip_resizes_from_equity_with_unrealized_pnl_before_close(self) -> None:
        client = DummyKrakenClient(position_signed=9.0, equity_usd=111.0, mark_price=100.0)
        res = execute_kraken_tv_signal(_signal("flip", "sell"), _config(dry_run=True), client)

        self.assertEqual(res["target_size"], 9.9)
        self.assertEqual(res["desired_signed"], -9.9)
        self.assertEqual(res["order_side"], "sell")
        self.assertEqual(res["order_size"], 18.9)
        self.assertEqual(client.market_orders, [])

    def test_tp1_closes_configured_fraction_reduce_only(self) -> None:
        client = DummyKrakenClient(position_signed=-9.0)
        res = execute_kraken_tv_signal(_signal("tp1", "", bar_index=2), _config(dry_run=False), client)

        self.assertTrue(res["ok"])
        self.assertEqual(res["order_side"], "buy")
        self.assertEqual(res["order_size"], 4.5)
        self.assertEqual(client.market_orders, [{"side": "buy", "size": 4.5, "symbol": "PF_SOLUSD", "reduce_only": True}])
        self.assertEqual(client.position_signed, -4.5)

    def test_dedupe_rejects_same_payload_inside_ttl(self) -> None:
        client = DummyKrakenClient(position_signed=0.0)
        sig = _signal("entry", "buy")
        config = _config(dry_run=False, dedup_ttl_sec=300)

        first = execute_kraken_tv_signal(sig, config, client)
        second = execute_kraken_tv_signal(sig, config, client)

        self.assertFalse(first.get("deduped", False))
        self.assertTrue(second["deduped"])
        self.assertEqual(len(client.market_orders), 1)

    def test_execution_failure_releases_dedupe_fingerprint_for_retry(self) -> None:
        class FailingClient(DummyKrakenClient):
            def get_mark_price(self, symbol=None) -> float:
                raise RuntimeError("temporary kraken failure")

        sig = _signal("entry", "buy")
        config = _config(dry_run=False, dedup_ttl_sec=300)

        with self.assertRaises(RuntimeError):
            execute_kraken_tv_signal(sig, config, FailingClient())

        retry = execute_kraken_tv_signal(sig, config, DummyKrakenClient())
        self.assertFalse(retry.get("deduped", False))

    def test_unsupported_exit_actions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _signal("sl", "", bar_index=3)

    def test_same_side_retarget_does_not_cancel_reduce_only_orders(self) -> None:
        client = DummyKrakenClient(position_signed=4.5, equity_usd=100.0, mark_price=100.0)
        res = execute_kraken_tv_signal(_signal("entry", "buy"), _config(dry_run=False), client)

        self.assertEqual(res["order_side"], "buy")
        self.assertEqual(res["order_size"], 4.5)
        self.assertEqual(client.cancel_calls, [])


class KrakenTVWebhookTests(unittest.TestCase):
    def test_endpoint_accepts_pine_payload_body_token(self) -> None:
        from quant.execution import webhook_server as ws

        class DummyRequest:
            async def json(self):
                return {
                    "token": "secret",
                    "action": "flip",
                    "side": "sell",
                    "symbol": "SOL-USDT",
                    "source": "test",
                    "reason": "unit",
                    "bar_time": 123,
                    "bar_index": 1,
                }

        with patch.dict("os.environ", {"WEBHOOK_TOKEN": "secret", "KRAKEN_TV_DRY_RUN": "1"}, clear=False), patch(
            "quant.execution.kraken_tv_executor.execute_kraken_tv_signal",
            return_value={"ok": True, "dry_run": True},
        ):
            body = asyncio.run(
                ws.kraken_tv_execute_webhook(
                    DummyRequest(),  # type: ignore[arg-type]
                    x_webhook_token=None,
                )
            )

        self.assertEqual(body, {"ok": True, "dry_run": True})


if __name__ == "__main__":
    unittest.main()
