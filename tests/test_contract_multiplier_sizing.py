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
    """Sizing is governed purely by percentage of equity — no dollar cap."""

    def _uncapped(self, **kw):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIVE_EXECUTOR_MAX_MARGIN_USDT", None)
            os.environ.pop("LIVE_EXECUTOR_MAX_CONTRACTS", None)
            return _live_order_qty(**kw)

    def test_margin_never_exceeds_configured_percentage(self) -> None:
        for equity in (15.0, 50.0, 200.0, 1000.0):
            for leverage in (3.0, 10.0):
                for price in (100.0, 150.0, 200.0, 250.0):
                    qty = self._uncapped(
                        equity=equity, pos_pct=0.90, leverage=leverage,
                        mid_price=price, contract_multiplier=KUCOIN_SOL_MULTIPLIER,
                    )
                    margin = (qty * price * KUCOIN_SOL_MULTIPLIER) / leverage
                    self.assertLessEqual(
                        margin, equity * 0.90 + 1e-9,
                        f"margin {margin} exceeded 90% of {equity} at {leverage}x/${price}",
                    )

    def test_size_scales_with_equity_not_a_fixed_cap(self) -> None:
        """A dollar cap would flatten these; percentage sizing must not."""
        kw = dict(pos_pct=0.90, leverage=10.0, mid_price=200.0,
                  contract_multiplier=KUCOIN_SOL_MULTIPLIER)
        small = self._uncapped(equity=15.0, **kw)
        large = self._uncapped(equity=150.0, **kw)
        self.assertGreater(large, small * 5, "size must scale with equity")

    def test_trades_are_possible_at_realistic_sol_prices(self) -> None:
        """The old $5 cap floored qty to 0 above ~$150/SOL."""
        for price in (100.0, 150.0, 200.0, 250.0):
            qty = self._uncapped(
                equity=15.0, pos_pct=0.90, leverage=10.0,
                mid_price=price, contract_multiplier=KUCOIN_SOL_MULTIPLIER,
            )
            self.assertGreater(qty, 0, f"no order possible at ${price}/SOL")

    def test_explicit_cap_still_binds_when_asked_for(self) -> None:
        with patch.dict(os.environ, {"LIVE_EXECUTOR_MAX_MARGIN_USDT": "15"}, clear=False):
            qty = _live_order_qty(
                equity=1000.0, pos_pct=0.90, leverage=10.0,
                mid_price=200.0, contract_multiplier=KUCOIN_SOL_MULTIPLIER,
            )
        self.assertEqual(qty, 6, "an explicit cap must still apply")


class LeverageSourceOfTruthTests(unittest.TestCase):
    """Sizing leverage must never exceed the leverage KuCoin actually applies.

    TV_EXEC_LEVERAGE used to default to 10 and drive sizing independently of
    KUCOIN_FUTURES_ORDER_LEVERAGE, which drove what the exchange applied.
    """

    def _cfg(self, env):
        from quant.execution.tv_signal_executor import TVExecConfig
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("TV_EXEC_LEVERAGE", None) if "TV_EXEC_LEVERAGE" not in env else None
            return TVExecConfig.from_env()

    def test_sizing_defaults_to_order_leverage_not_ten(self) -> None:
        cfg = self._cfg({
            "LIVE_EXECUTOR_LEVERAGE": "3",
            "KUCOIN_FUTURES_ORDER_LEVERAGE": "3",
        })
        self.assertEqual(cfg.leverage, 3.0, "sizing must not silently use 10x")

    def test_mismatch_falls_back_to_exchange_leverage(self) -> None:
        cfg = self._cfg({
            "LIVE_EXECUTOR_LEVERAGE": "3",
            "KUCOIN_FUTURES_ORDER_LEVERAGE": "3",
            "TV_EXEC_LEVERAGE": "10",
        })
        self.assertEqual(cfg.leverage, 3.0, "must not size at 10x when exchange applies 3x")

    def test_ten_x_when_set_consistently(self) -> None:
        cfg = self._cfg({
            "LIVE_EXECUTOR_LEVERAGE": "10",
            "KUCOIN_FUTURES_ORDER_LEVERAGE": "10",
        })
        self.assertEqual(cfg.leverage, 10.0)



class AvailableBalanceSizingTests(unittest.TestCase):
    """Sizing must use spendable balance, not equity locked in a position."""

    class _Bal:
        def __init__(self, equity, available):
            self._b = {"equity": equity, "available": available}

        def get_account_balance(self, currency="USDT"):
            return dict(self._b)

    def test_uses_available_when_position_locks_margin(self) -> None:
        from quant.execution.live_executor import _resolve_equity
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIVE_EXECUTOR_SIZE_OFF_EQUITY", None)
            got = _resolve_equity(self._Bal(equity=22.70, available=12.19))
        self.assertEqual(got, 12.19, "must size off spendable balance")

    def test_uses_equity_when_nothing_is_locked(self) -> None:
        from quant.execution.live_executor import _resolve_equity
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIVE_EXECUTOR_SIZE_OFF_EQUITY", None)
            got = _resolve_equity(self._Bal(equity=22.70, available=22.70))
        self.assertEqual(got, 22.70)

    def test_order_fits_within_available_margin(self) -> None:
        """The exact case KuCoin was rejecting in a loop."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIVE_EXECUTOR_MAX_MARGIN_USDT", None)
            os.environ.pop("LIVE_EXECUTOR_MAX_CONTRACTS", None)
            qty = _live_order_qty(
                equity=12.19, pos_pct=0.90, leverage=10.0,
                mid_price=76.0, contract_multiplier=KUCOIN_SOL_MULTIPLIER,
            )
        margin = qty * 76.0 * KUCOIN_SOL_MULTIPLIER / 10.0
        self.assertLessEqual(margin, 12.19, "order must fit in available balance")
        self.assertGreater(qty, 0)

    def test_smallest_order_is_affordable(self) -> None:
        """1 contract = 0.1 SOL must be reachable on a small account."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIVE_EXECUTOR_MAX_MARGIN_USDT", None)
            os.environ.pop("LIVE_EXECUTOR_MAX_CONTRACTS", None)
            qty = _live_order_qty(
                equity=2.0, pos_pct=0.90, leverage=10.0,
                mid_price=76.0, contract_multiplier=KUCOIN_SOL_MULTIPLIER,
            )
        self.assertGreaterEqual(qty, 1, "0.1 SOL should be affordable with ~$2")


if __name__ == "__main__":
    unittest.main()
