"""Read-only KuCoin Hedge Mode preflight; no real exchange calls."""
import unittest

from quant.execution.family_hedge_preflight import (
    HedgePreflightError,
    inspect_hedge_account,
    require_hedge_ready,
)


class FakeBroker:
    def __init__(self, mode=1, positions=(), orders=(), stops=()):
        self.mode = mode
        self.positions = list(positions)
        self.orders = list(orders)
        self.stops = list(stops)
        self.methods = []

    def _req(self, method, path, body=None):
        self.methods.append((method, path))
        if method != "GET":
            raise AssertionError("preflight may only READ")
        if path == "/api/v2/position/getPositionMode":
            return {"positionMode": self.mode}
        if path == "/api/v1/positions":
            return self.positions
        raise AssertionError(f"unexpected endpoint {path}")

    def list_open_orders(self, symbol):
        self.methods.append(("GET", "open-orders"))
        return self.orders

    def list_open_stop_orders(self, symbol):
        self.methods.append(("GET", "stop-orders"))
        return self.stops


class HedgePreflightTests(unittest.TestCase):
    def test_both_sides_and_open_orders_are_visible(self):
        b = FakeBroker(
            positions=[
                {"symbol": "SOLUSDTM", "positionSide": "LONG", "currentQty": 3},
                {"symbol": "SOLUSDTM", "positionSide": "SHORT", "currentQty": -4},
            ],
            orders=[{"clientOid": "order1"}],
            stops=[{"clientOid": "stop1"}],
        )
        state = inspect_hedge_account(b)
        self.assertEqual(state.mode, "HEDGE")
        self.assertEqual((state.long_contracts, state.short_contracts), (3, 4))
        self.assertEqual(state.open_regular_orders, 1)
        self.assertEqual(state.open_conditional_orders, 1)
        self.assertFalse(state.flat_and_order_free)
        require_hedge_ready(state)
        self.assertTrue(all(method == "GET" for method, _ in b.methods))

    def test_empty_account_reports_flat(self):
        s = inspect_hedge_account(FakeBroker())
        self.assertTrue(s.flat_and_order_free)
        self.assertTrue(s.hedge_ready)

    def test_one_way_is_blocked(self):
        s = inspect_hedge_account(FakeBroker(mode=0))
        with self.assertRaises(HedgePreflightError):
            require_hedge_ready(s)

    def test_nonflat_both_position_fails_closed(self):
        b = FakeBroker(positions=[
            {"symbol": "SOLUSDTM", "positionSide": "BOTH", "currentQty": 5},
        ])
        with self.assertRaises(HedgePreflightError):
            inspect_hedge_account(b)

    def test_missing_positions_api_must_not_look_flat(self):
        b = FakeBroker()
        b.positions = None
        with self.assertRaises(HedgePreflightError):
            inspect_hedge_account(b)

    def test_other_symbol_position_blocks_dedicated_account(self):
        b = FakeBroker(positions=[
            {"symbol": "XBTUSDTM", "positionSide": "LONG", "currentQty": 1},
        ])
        with self.assertRaises(HedgePreflightError):
            inspect_hedge_account(b)

    def test_malformed_contract_quantity_fails(self):
        b = FakeBroker(positions=[
            {"symbol": "SOLUSDTM", "positionSide": "LONG", "currentQty": "nan"},
        ])
        with self.assertRaises(HedgePreflightError):
            inspect_hedge_account(b)


if __name__ == "__main__":
    unittest.main()
