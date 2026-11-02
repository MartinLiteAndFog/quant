"""Regression checks for independent Blaupunkt hedge strategy books."""
import pytest

from quant.execution.blaupunkt_hedge import HedgeFleet


class FakeBroker:
    def __init__(self):
        self.positions = {"long": 0, "short": 0}
        self.orders = {}
        self.seq = 0
        self.cancelled = []

    def _req(self, method, path, body=None):
        if method == "GET" and "getPositionMode" in path:
            return {"positionMode": 1}
        if method == "GET" and path.startswith("/api/v2/position?"):
            return [
                {"symbol": "XBTUSDTM", "positionSide": side.upper(), "currentQty": qty}
                for side, qty in self.positions.items() if qty
            ]
        if method == "GET" and path.startswith("/api/v1/contracts/"):
            return {"tickSize": 0.1}
        if method == "POST" and path == "/api/v1/orders":
            assert body["positionSide"] in ("LONG", "SHORT")
            self.seq += 1
            oid = str(self.seq)
            side = body["positionSide"].lower()
            qty = int(body["size"])
            stop = body.get("stop")
            if stop:
                self.orders[oid] = {"status": "open", "isActive": True,
                                    "stop": stop, "filledSize": 0}
            else:
                self.positions[side] += -qty if body["reduceOnly"] else qty
                assert self.positions[side] >= 0
                self.orders[oid] = {"status": "done", "isActive": False,
                                    "filledSize": qty}
            return {"orderId": oid}
        raise AssertionError((method, path, body))

    def get_order(self, oid):
        return dict(self.orders[str(oid)])

    def cancel_order(self, order_id=None, client_id=None):
        self.cancelled.append(str(order_id))
        self.orders[str(order_id)]["isActive"] = False
        self.orders[str(order_id)]["status"] = "done"

    def get_account_balance(self, currency="USDT"):
        return {"equity": 1000, "available": 1000}

    def get_best_bid_ask(self, symbol):
        return 100000, 100000

    def get_contract_multiplier(self, symbol):
        return 0.001


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    for name, val in {
        "HEDGE_BOT_IDS": "Blaupunkt,ElliotP1",
        "HEDGE_SYMBOL": "XBT-USDT",
        "HEDGE_AUTO_REGISTER": "1",
        "HEDGE_BOT_CAPITAL_USAGE": "1",
        "HEDGE_MAX_LEVERAGE": "20",
        "KUCOIN_FUTURES_ORDER_LEVERAGE": "10",
        "KUCOIN_FUTURES_MARGIN_MODE": "ISOLATED",
        "LIVE_TRADING_ENABLED": "1",
    }.items():
        monkeypatch.setenv(name, val)
    return HedgeFleet(broker=FakeBroker(), path=tmp_path / "ledger.json")


def alert(fleet, bot, action, side="", symbol="XBT-USDT", dry_run=False):
    return fleet.execute(bot, action, side, symbol, None, 0.5, dry_run)


def test_two_bots_long_and_short_independent(fleet):
    a = alert(fleet, "Blaupunkt", "flip", "buy")
    b = alert(fleet, "ElliotP1", "flip", "sell")
    assert a["qty"] == b["qty"] == 50
    assert fleet.broker.positions == {"long": 50, "short": 50}
    assert fleet.status()["allocation_pct"] == 50
    alert(fleet, "Blaupunkt", "exit")
    assert fleet.broker.positions == {"long": 0, "short": 50}
    assert fleet.status()["bots"][1]["side"] == "short"


def test_flip_one_bot_does_not_flatten_other(fleet):
    alert(fleet, "Blaupunkt", "flip", "buy")
    alert(fleet, "ElliotP1", "flip", "buy")
    alert(fleet, "Blaupunkt", "flip", "sell")
    assert fleet.broker.positions == {"long": 50, "short": 50}
    assert fleet.status()["bots"][0]["side"] == "short"
    assert fleet.status()["bots"][1]["side"] == "long"


def test_auto_registration_changes_only_new_order_sizing(fleet):
    alert(fleet, "Blaupunkt", "flip", "buy")
    c = alert(fleet, "ThirdBot", "flip", "sell")
    assert c["bot_count"] == 3
    assert c["qty"] == 33
    assert fleet.broker.positions == {"long": 50, "short": 33}
    assert fleet.status()["allocation_pct"] == pytest.approx(100 / 3)


def test_tp1_scoped_to_owner_and_protection_replaced(fleet):
    alert(fleet, "Blaupunkt", "flip", "buy")
    alert(fleet, "ElliotP1", "flip", "buy")
    old_stop = fleet._read()["bots"]["blaupunkt"]["stop_id"]
    result = alert(fleet, "Blaupunkt", "tp1")
    assert result["remaining"] == 25
    assert fleet.broker.positions["long"] == 75
    assert old_stop in fleet.broker.cancelled
    assert fleet._read()["bots"]["blaupunkt"]["stop_id"] != old_stop


def test_reconciliation_fails_closed(fleet):
    alert(fleet, "Blaupunkt", "flip", "buy")
    fleet.broker.positions["long"] += 1  # Manual/external trade
    with pytest.raises(RuntimeError, match="reconciliation mismatch"):
        alert(fleet, "ElliotP1", "flip", "sell")
    assert fleet.broker.positions["short"] == 0


def test_wrong_symbol_or_id_rejected(fleet):
    with pytest.raises(ValueError, match="symbol"):
        alert(fleet, "Blaupunkt", "flip", "buy", symbol="SOL-USDT")
    with pytest.raises(ValueError, match="bot ID"):
        alert(fleet, "some spaces", "flip", "buy")


def test_dry_run_does_not_change_exchange(fleet):
    result = alert(fleet, "Blaupunkt", "flip", "buy", dry_run=True)
    assert result["dry_run"]
    assert fleet.broker.positions == {"long": 0, "short": 0}
    assert fleet._read()["bots"]["blaupunkt"]["qty"] == 0


def test_state_persists_across_restart(fleet):
    alert(fleet, "Blaupunkt", "flip", "buy")
    again = HedgeFleet(broker=fleet.broker, path=fleet.path)
    assert again.status()["bots"][0]["qty"] == 50
    assert alert(again, "Blaupunkt", "flip", "buy")["reason"] == "duplicate_same_side"


def test_non_hedge_account_no_trade(fleet):
    original = fleet.broker._req
    def oneway(method, path, body=None):
        if "getPositionMode" in path:
            return {"positionMode": 0}
        return original(method, path, body)
    fleet.broker._req = oneway
    with pytest.raises(RuntimeError, match="Hedge Mode"):
        alert(fleet, "Blaupunkt", "flip", "buy")
    assert fleet.broker.positions == {"long": 0, "short": 0}
