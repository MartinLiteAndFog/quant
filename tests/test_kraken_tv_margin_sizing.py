"""Tests for margin-aware sizing of the Kraken fallback reopen.

Incident 2026-07-25 17:45Z: a `flip/buy` on PF_SOLUSD closed the existing
position, then every attempt to open the new one was rejected by Kraken with
`insufficientavailablefunds` (5 retries, all HTTP 500 back to TradingView). The
account was left flat and the entry was missed.

Cause: closing frees margin asynchronously. The position reads flat while its
collateral is still locked, so `_wait_for_flat` returns, sizing is computed from
*total equity*, and the resulting order needs more margin than is actually free.

  equity $482.19, available $403.37, mark 74.40, pos_pct 0.90, leverage 10x
  -> target 58.3 SOL, notional $4337.52, margin required $433.75
  -> $433.75 > $403.37 available, short by $30.38 -> rejected

The fix waits for the collateral to come back, then caps the order to what free
margin can actually back, and retries a rejected open at a fitted size.
"""

from __future__ import annotations

import unittest

from quant.execution.kraken_tv_executor import (
    KrakenTVConfig,
    _is_insufficient_funds,
    _margin_required_usd,
    _place_open_with_margin_retry,
    _size_within_available,
    _wait_for_available_margin,
    compute_target_size,
)

# The exact state at the time of the incident.
EQUITY = 482.19
AVAILABLE_AT_FAILURE = 403.37
MARK = 74.40
POS_PCT = 0.90
LEVERAGE = 10.0
STEP = 0.1


def _config(**overrides) -> KrakenTVConfig:
    base = dict(
        venue_symbol="PF_SOLUSD",
        display_symbol="SOL-USDT",
        pos_pct=POS_PCT,
        leverage=LEVERAGE,
        tp1_frac=0.5,
        dry_run=False,
        size_step=STEP,
        dedup_ttl_sec=300.0,
        cancel_reduce_only_on_flip=True,
        verify_after_order=True,
        refill_partial=False,
        margin_buffer=0.98,
        margin_wait_attempts=3,
        margin_wait_delay_sec=0.0,
        open_retry_attempts=3,
    )
    base.update(overrides)
    return KrakenTVConfig(**base)


class _FakeClient:
    """Minimal stand-in: a scripted sequence of available-margin readings."""

    def __init__(self, available_sequence, reject_below=None):
        self._available = list(available_sequence)
        self._reject_below = reject_below
        self.orders = []

    def get_account_equity(self):
        value = self._available[0] if len(self._available) == 1 else self._available.pop(0)
        return {"equity_usd": EQUITY, "available_usd": value, "wallet_usd": EQUITY}

    def place_market(self, side, *, size, symbol, reduce_only):
        self.orders.append({"side": side, "size": size, "reduce_only": reduce_only})
        if self._reject_below is not None and _margin_required_usd(size, MARK, LEVERAGE) > self._reject_below:
            return {"ok": False, "sendStatus": {"status": "insufficientavailablefunds"}}
        return {"ok": True, "sendStatus": {"status": "placed"}}


class TestIncidentArithmetic(unittest.TestCase):
    def test_old_behaviour_would_have_been_rejected(self) -> None:
        target = compute_target_size(
            equity_usd=EQUITY, mark_price=MARK, leverage=LEVERAGE, pos_pct=POS_PCT, step=STEP
        )
        self.assertAlmostEqual(target, 58.3, places=1)
        required = _margin_required_usd(target, MARK, LEVERAGE)
        self.assertGreater(required, AVAILABLE_AT_FAILURE)  # this is the bug

    def test_new_sizing_fits_the_available_margin(self) -> None:
        size = _size_within_available(
            equity_usd=EQUITY,
            available_usd=AVAILABLE_AT_FAILURE,
            mark_price=MARK,
            leverage=LEVERAGE,
            pos_pct=POS_PCT,
            step=STEP,
            buffer=0.98,
        )
        self.assertGreater(size, 0.0, "must still open a position, not give up")
        self.assertLessEqual(_margin_required_usd(size, MARK, LEVERAGE), AVAILABLE_AT_FAILURE)


class TestSizeWithinAvailable(unittest.TestCase):
    def test_full_margin_available_leaves_target_untouched(self) -> None:
        target = compute_target_size(
            equity_usd=EQUITY, mark_price=MARK, leverage=LEVERAGE, pos_pct=POS_PCT, step=STEP
        )
        size = _size_within_available(
            equity_usd=EQUITY, available_usd=EQUITY, mark_price=MARK, leverage=LEVERAGE,
            pos_pct=POS_PCT, step=STEP, buffer=0.98,
        )
        self.assertEqual(size, target)

    def test_never_exceeds_the_configured_target(self) -> None:
        # Even with absurd free collateral, sizing stays at the configured target
        # — the cap may only shrink, never grow, the position.
        target = compute_target_size(
            equity_usd=EQUITY, mark_price=MARK, leverage=LEVERAGE, pos_pct=POS_PCT, step=STEP
        )
        size = _size_within_available(
            equity_usd=EQUITY, available_usd=EQUITY * 100, mark_price=MARK, leverage=LEVERAGE,
            pos_pct=POS_PCT, step=STEP, buffer=0.98,
        )
        self.assertEqual(size, target)

    def test_never_returns_zero_for_a_nonzero_target(self) -> None:
        # Opening nothing is the bug being fixed. If free collateral cannot fund
        # even one step, fall back to the intended target and let Kraken reject
        # it — a visible, retryable rejection beats a silent no-op.
        target = compute_target_size(
            equity_usd=EQUITY, mark_price=MARK, leverage=LEVERAGE, pos_pct=POS_PCT, step=STEP
        )
        size = _size_within_available(
            equity_usd=EQUITY, available_usd=0.0, mark_price=MARK, leverage=LEVERAGE,
            pos_pct=POS_PCT, step=STEP, buffer=0.98,
        )
        self.assertEqual(size, target)

    def test_unknown_available_does_not_cap(self) -> None:
        # A venue that does not report free collateral must not be treated as
        # broke; preserve the pre-fix sizing rather than refusing to trade.
        target = compute_target_size(
            equity_usd=EQUITY, mark_price=MARK, leverage=LEVERAGE, pos_pct=POS_PCT, step=STEP
        )
        size = _size_within_available(
            equity_usd=EQUITY, available_usd=None, mark_price=MARK, leverage=LEVERAGE,
            pos_pct=POS_PCT, step=STEP, buffer=0.98,
        )
        self.assertEqual(size, target)


class TestWaitForAvailableMargin(unittest.TestCase):
    def test_returns_as_soon_as_margin_is_released(self) -> None:
        client = _FakeClient([AVAILABLE_AT_FAILURE, 420.0, EQUITY])
        got = _wait_for_available_margin(client, 433.75, attempts=3, delay_sec=0.0)
        self.assertGreaterEqual(got, 433.75)

    def test_gives_up_and_reports_last_reading(self) -> None:
        client = _FakeClient([AVAILABLE_AT_FAILURE])
        got = _wait_for_available_margin(client, 433.75, attempts=3, delay_sec=0.0)
        self.assertAlmostEqual(got, AVAILABLE_AT_FAILURE)


class TestInsufficientFundsDetection(unittest.TestCase):
    def test_matches_the_real_kraken_message(self) -> None:
        exc = RuntimeError(
            "kraken tv fallback open rejected by Kraken: "
            "unexpected order status: insufficientavailablefunds"
        )
        self.assertTrue(_is_insufficient_funds(exc))

    def test_does_not_match_unrelated_rejections(self) -> None:
        self.assertFalse(_is_insufficient_funds(RuntimeError("invalid size")))
        self.assertFalse(_is_insufficient_funds(RuntimeError("marketSuspended")))


class TestPlaceOpenWithMarginRetry(unittest.TestCase):
    def test_shrinks_and_succeeds_after_a_funds_rejection(self) -> None:
        # Kraken only accepts orders needing <= $403.37 of margin.
        client = _FakeClient([AVAILABLE_AT_FAILURE], reject_below=AVAILABLE_AT_FAILURE)
        result, size = _place_open_with_margin_retry(
            client, _config(), side="buy", size=58.3, mark_price=MARK, equity_usd=EQUITY,
        )
        self.assertIsNotNone(result)
        self.assertGreater(size, 0.0)
        self.assertLess(size, 58.3, "should have shrunk to fit")
        self.assertLessEqual(_margin_required_usd(size, MARK, LEVERAGE), AVAILABLE_AT_FAILURE)

    def test_first_attempt_succeeds_when_funds_are_there(self) -> None:
        client = _FakeClient([EQUITY], reject_below=None)
        result, size = _place_open_with_margin_retry(
            client, _config(), side="buy", size=58.3, mark_price=MARK, equity_usd=EQUITY,
        )
        self.assertIsNotNone(result)
        self.assertEqual(size, 58.3)
        self.assertEqual(len(client.orders), 1, "no needless retry")

    def test_never_grows_the_order_on_retry(self) -> None:
        client = _FakeClient([EQUITY * 100], reject_below=AVAILABLE_AT_FAILURE)
        try:
            _, size = _place_open_with_margin_retry(
                client, _config(), side="buy", size=30.0, mark_price=MARK, equity_usd=EQUITY,
            )
        except RuntimeError:
            size = 0.0
        for order in client.orders:
            self.assertLessEqual(order["size"], 30.0)

    def test_non_funds_rejection_propagates_immediately(self) -> None:
        class _Rejecting(_FakeClient):
            def place_market(self, side, *, size, symbol, reduce_only):
                self.orders.append({"size": size})
                return {"ok": False, "sendStatus": {"status": "rejected", "rejectReason": "marketSuspended"}}

        client = _Rejecting([EQUITY])
        with self.assertRaises(RuntimeError):
            _place_open_with_margin_retry(
                client, _config(), side="buy", size=58.3, mark_price=MARK, equity_usd=EQUITY,
            )
        self.assertEqual(len(client.orders), 1, "must not retry a non-funds rejection")

    def test_surfaces_the_rejection_when_genuinely_broke(self) -> None:
        # No collateral at all: attempts still happen (so the failure is visible
        # to TradingView and the logs) and the error propagates rather than the
        # bot quietly sitting flat.
        client = _FakeClient([0.0], reject_below=0.0)
        with self.assertRaises(RuntimeError):
            _place_open_with_margin_retry(
                client, _config(), side="buy", size=58.3, mark_price=MARK, equity_usd=EQUITY,
            )
        self.assertGreater(len(client.orders), 0, "must have actually tried")

    def test_missing_available_field_still_places_the_order(self) -> None:
        # Regression guard: an account payload without `available_usd` once made
        # sizing collapse to 0 and skip the reopen entirely.
        class _NoAvailable(_FakeClient):
            def get_account_equity(self):
                return {"equity_usd": EQUITY, "wallet_usd": EQUITY}

        client = _NoAvailable([EQUITY])
        result, size = _place_open_with_margin_retry(
            client, _config(), side="buy", size=58.3, mark_price=MARK, equity_usd=EQUITY,
        )
        self.assertIsNotNone(result)
        self.assertEqual(size, 58.3)


if __name__ == "__main__":
    unittest.main()
