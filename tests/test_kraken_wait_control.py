from __future__ import annotations

import unittest
from unittest.mock import patch

from quant.execution.kraken_wait_control import (
    clear_wait_mode_pin,
    get_wait_mode_pin,
    reconcile_wait_mode_pin,
    set_wait_mode_pin,
    wait_mode_pin_key,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.last_set_ex = None

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str, ex=None):
        self.store[key] = value
        self.last_set_ex = ex
        return True

    def delete(self, key: str):
        existed = key in self.store
        self.store.pop(key, None)
        return 1 if existed else 0


class KrakenWaitControlTests(unittest.TestCase):
    def test_set_get_clear_roundtrip(self) -> None:
        fake = _FakeRedis()
        with (
            patch("quant.execution.kraken_wait_control._require_redis_client", return_value=fake),
            patch("quant.execution.kraken_wait_control._maybe_redis_client", return_value=fake),
        ):
            payload = set_wait_mode_pin(
                "SOL-USDT",
                side="long",
                reason="manual_trade",
                actor="test",
                ttl_sec=300,
            )
            loaded = get_wait_mode_pin("SOL-USDT")
            cleared = clear_wait_mode_pin("SOL-USDT")

        self.assertEqual(payload["key"], wait_mode_pin_key("SOL-USDT"))
        self.assertEqual(payload["side"], "long")
        self.assertEqual(payload["reason"], "manual_trade")
        self.assertEqual(fake.last_set_ex, 300)
        self.assertIsInstance(loaded, dict)
        self.assertEqual(loaded["actor"], "test")
        self.assertTrue(cleared["cleared"])

    def test_reconcile_keeps_matching_side_pin(self) -> None:
        pin = {"symbol": "SOL-USDT", "side": "long", "reason": "manual_trade"}
        with (
            patch("quant.execution.kraken_wait_control.get_wait_mode_pin", return_value=pin),
            patch("quant.execution.kraken_wait_control.clear_wait_mode_pin") as clear_mock,
        ):
            active, payload = reconcile_wait_mode_pin("SOL-USDT", "long")

        self.assertTrue(active)
        self.assertEqual(payload, pin)
        clear_mock.assert_not_called()

    def test_reconcile_clears_stale_pin_after_close(self) -> None:
        pin = {"symbol": "SOL-USDT", "side": "long", "reason": "manual_trade"}
        with (
            patch("quant.execution.kraken_wait_control.get_wait_mode_pin", return_value=pin),
            patch(
                "quant.execution.kraken_wait_control.clear_wait_mode_pin",
                return_value={"symbol": "SOL-USDT", "cleared": True},
            ) as clear_mock,
        ):
            active, payload = reconcile_wait_mode_pin("SOL-USDT", "flat")

        self.assertFalse(active)
        self.assertEqual(payload, pin)
        clear_mock.assert_called_once_with("SOL-USDT")
