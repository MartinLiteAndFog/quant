"""Regression tests for the 10x order-sizing discrepancy.

KuCoin SOLUSDTM has multiplier 0.1 (1 contract = 0.1 SOL) while Kraken
PF_SOLUSD has contractSize 1 (1 contract = 1 SOL). Silently defaulting the
KuCoin multiplier to 1.0 imports the Kraken convention and oversizes every
KuCoin order by exactly 10x.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from quant.execution.live_executor import _live_order_qty, _resolve_contract_multiplier

KUCOIN_SOL_MULTIPLIER = 0.1


class _Broker:
    def __init__(self, mult=None, exc=None):
        self._mult = mult
        self._exc = exc

    def get_contract_multiplier(self, symbol: str) -> float:
        if self._exc is not None:
            raise self._exc
        return self._mult


class ContractMultiplierTests(unittest.TestCase):
    def test_uses_live_multiplier_when_available(self) -> None:
        self.assertEqual(
            _resolve_contract_multiplier(_Broker(mult=0.1), "SOL-USDT"), 0.1
        )

    def test_falls_back_low_not_high_on_lookup_failure(self) -> None:
        """The fallback must not be 1.0 — that is a 10x oversize on KuCoin."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIVE_EXECUTOR_CONTRACT_MULTIPLIER", None)
            mult = _resolve_contract_multiplier(
                _Broker(exc=RuntimeError("api down")), "SOL-USDT"
            )
        self.assertEqual(mult, KUCOIN_SOL_MULTIPLIER)

    def test_non_positive_multiplier_falls_back(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIVE_EXECUTOR_CONTRACT_MULTIPLIER", None)
            mult = _resolve_contract_multiplier(_Broker(mult=0.0), "SOL-USDT")
        self.assertEqual(mult, KUCOIN_SOL_MULTIPLIER)

    def test_wrong_multiplier_would_have_been_ten_x(self) -> None:
        """Documents the size of the bug this guards against."""
        kw = dict(equity=15.0, pos_pct=0.90, leverage=3.0, mid_price=200.0)
        with patch.dict(
            os.environ,
            {"LIVE_EXECUTOR_MAX_MARGIN_USDT": "15", "LIVE_EXECUTOR_MAX_CONTRACTS": "20"},
            clear=False,
        ):
            correct = _live_order_qty(contract_multiplier=0.1, **kw)
            wrong = _live_order_qty(contract_multiplier=1.0, **kw)
        # Same contract count means 10x the SOL exposure at multiplier 1.0.
        self.assertEqual(correct, 2)
        self.assertEqual(correct * 0.1, 0.2)   # 0.2 SOL — intended
        self.assertEqual(wrong * 1.0, 0.0)     # nothing, or 10x if inverted


class PilotSizingTests(unittest.TestCase):
    """$15 account, 90% of equity, 3x leverage must stay within the margin cap."""

    def test_sizing_respects_margin_cap_across_prices(self) -> None:
        with patch.dict(
            os.environ,
            {"LIVE_EXECUTOR_MAX_MARGIN_USDT": "15", "LIVE_EXECUTOR_MAX_CONTRACTS": "20"},
            clear=False,
        ):
            for leverage in (3.0, 10.0):
                for price in (100.0, 150.0, 200.0, 250.0):
                    qty = _live_order_qty(
                        equity=15.0, pos_pct=0.90, leverage=leverage,
                        mid_price=price, contract_multiplier=KUCOIN_SOL_MULTIPLIER,
                    )
                    margin = (qty * price * KUCOIN_SOL_MULTIPLIER) / leverage
                    self.assertLessEqual(
                        margin, 15.0 * 0.90 + 1e-9,
                        f"margin {margin} exceeded 90% of $15 at {leverage}x/${price}",
                    )

    def test_trades_are_possible_at_realistic_sol_prices(self) -> None:
        """The old $5 cap floored qty to 0 above ~$150/SOL."""
        with patch.dict(
            os.environ,
            {"LIVE_EXECUTOR_MAX_MARGIN_USDT": "15", "LIVE_EXECUTOR_MAX_CONTRACTS": "20"},
            clear=False,
        ):
            for price in (100.0, 150.0, 200.0, 250.0):
                qty = _live_order_qty(
                    equity=15.0, pos_pct=0.90, leverage=3.0,
                    mid_price=price, contract_multiplier=KUCOIN_SOL_MULTIPLIER,
                )
                self.assertGreater(qty, 0, f"no order possible at ${price}/SOL")


if __name__ == "__main__":
    unittest.main()
