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
        if method == "GET" and path == "/api/v1/positions":
            return [
                {"symbol": "XBTUSDTM", "currentQty": qty}
                for qty in self.positions.values() if qty
            ]
        if method == "GET" and (
            path.startswith("/api/v1/orders?status=active")
            or path.startswith("/api/v1/stopOrders?status=active")
        ):
            return {"totalNum": 0, "items": []}
        if method == "POST" and "switchPositionMode" in path:
            assert body == {"positionMode": "1"}
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
        return {"equity": 1000, "available": 1000 - sum(self.positions.values()) * 10}

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
        "HEDGE_AUTO_SET_POSITION_MODE": "0",
        "HEDGE_BOT_CAPITAL_USAGE": "0.90",
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
    assert a["qty"] == b["qty"] == 45
    assert fleet.broker.positions == {"long": 45, "short": 45}
    assert fleet.status()["allocation_pct"] == 45
    alert(fleet, "Blaupunkt", "exit")
    assert fleet.broker.positions == {"long": 0, "short": 45}
    assert fleet.status()["bots"][1]["side"] == "short"
    assert fleet.status()["capital_pool_pct"] == 90
    assert fleet.status()["reserve_pct"] == 10


def test_flip_one_bot_does_not_flatten_other(fleet):
    alert(fleet, "Blaupunkt", "flip", "buy")
    alert(fleet, "ElliotP1", "flip", "buy")
    alert(fleet, "Blaupunkt", "flip", "sell")
    assert fleet.broker.positions == {"long": 45, "short": 45}
    assert fleet.status()["bots"][0]["side"] == "short"
    assert fleet.status()["bots"][1]["side"] == "long"


def test_auto_registration_changes_only_new_order_sizing(fleet):
    alert(fleet, "Blaupunkt", "flip", "buy")
    c = alert(fleet, "ThirdBot", "flip", "sell")
    assert c["bot_count"] == 3
    assert c["qty"] == 30
    assert fleet.broker.positions == {"long": 45, "short": 30}
    assert fleet.status()["allocation_pct"] == 30


def test_tp1_scoped_to_owner_and_protection_replaced(fleet):
    alert(fleet, "Blaupunkt", "flip", "buy")
    alert(fleet, "ElliotP1", "flip", "buy")
    old_stop = fleet._read()["bots"]["blaupunkt"]["stop_id"]
    result = alert(fleet, "Blaupunkt", "tp1")
    assert result["remaining"] == 23
    assert fleet.broker.positions["long"] == 68
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
    assert again.status()["bots"][0]["qty"] == 45
    assert alert(again, "Blaupunkt", "flip", "buy")["reason"] == "duplicate_same_side"


def test_non_hedge_account_no_trade(fleet):
    original = fleet.broker._req
    def oneway(method, path, body=None):
        if "getPositionMode" in path:
            return {"positionMode": 0}
        return original(method, path, body)
    fleet.broker._req = oneway
    with pytest.raises(RuntimeError, match="auto-switch disabled"):
        alert(fleet, "Blaupunkt", "flip", "buy")
    assert fleet.broker.positions == {"long": 0, "short": 0}

def test_three_bot_pool_uses_ninety_percent_not_compounded(fleet):
    first = alert(fleet, "Blaupunkt", "flip", "buy")
    second = alert(fleet, "ElliotP1", "flip", "sell")
    # With N=2, 45% each. Close both then add a third registered bot.
    alert(fleet, "Blaupunkt", "exit")
    alert(fleet, "ElliotP1", "exit")
    assert first["qty"] == second["qty"] == 45
    assert fleet.broker.positions == {"long": 0, "short": 0}
    assert alert(fleet, "ThirdBot", "exit")["reason"] == "already_flat"
    # Even when available balance falls to 400 after two 300-USDT slots,
    # the third slot should receive 300, keeping exactly 100 reserve.
    a = alert(fleet, "Blaupunkt", "flip", "buy")
    b = alert(fleet, "ElliotP1", "flip", "sell")
    c = alert(fleet, "ThirdBot", "flip", "buy")
    assert [a["qty"], b["qty"], c["qty"]] == [30, 30, 30]
    assert fleet.broker.get_account_balance()["available"] == 100


def test_more_than_ninety_percent_pool_forbidden(fleet, monkeypatch):
    monkeypatch.setenv("HEDGE_BOT_CAPITAL_USAGE", "1.0")
    with pytest.raises(RuntimeError, match="capital usage"):
        alert(fleet, "Blaupunkt", "flip", "buy")
    assert fleet.broker.positions == {"long": 0, "short": 0}

def test_auto_switch_only_if_account_empty(fleet, monkeypatch):
    monkeypatch.setenv("HEDGE_AUTO_SET_POSITION_MODE", "1")
    mode = {"value": 0}
    original = fleet.broker._req
    def wrapper(method, path, body=None):
        if "getPositionMode" in path:
            return {"positionMode": mode["value"]}
        if method == "POST" and "switchPositionMode" in path:
            mode["value"] = 1
        return original(method, path, body)
    fleet.broker._req = wrapper
    assert fleet.ensure_hedge_mode() == {"hedge_mode": True, "changed": True}
    assert mode["value"] == 1


def test_auto_switch_rejects_any_outstanding_position(fleet, monkeypatch):
    monkeypatch.setenv("HEDGE_AUTO_SET_POSITION_MODE", "1")
    fleet.broker.positions["long"] = 1
    original = fleet.broker._req
    def wrapper(method, path, body=None):
        if "getPositionMode" in path:
            return {"positionMode": 0}
        return original(method, path, body)
    fleet.broker._req = wrapper
    with pytest.raises(RuntimeError, match="open positions"):
        fleet.ensure_hedge_mode()

def test_durable_alert_queue_deduplicates_and_executes(tmp_path, monkeypatch, fleet):
    from types import SimpleNamespace
    from quant.execution import hedge_webhook_queue as q
    from quant.execution.tv_signal_executor import TVSignal

    monkeypatch.setenv("HEDGE_WEBHOOK_QUEUE_FILE", str(tmp_path / "alerts.sqlite3"))
    monkeypatch.setattr(q, "fleet", lambda: fleet)
    signal = TVSignal(action="flip", side="buy", symbol="XBT-USDT")
    config = SimpleNamespace(tp1_close_pct=0.5, dry_run=False)
    first = q.enqueue("Blaupunkt", signal, config)
    duplicate = q.enqueue("Blaupunkt", signal, config)
    assert first["queued"]
    assert duplicate["duplicate"]
    assert duplicate["event_id"] == first["event_id"]
    assert q.status()["counts"] == {"queued": 1}
    assert q._process_next()
    assert q.status()["counts"] == {"executed": 1}
    assert fleet.broker.positions["long"] == 45
    assert q.status()["recent"][0]["result"]["bot"] == "Blaupunkt"


def test_queue_does_not_replay_interrupted_order(tmp_path, monkeypatch):
    from quant.execution import hedge_webhook_queue as q
    monkeypatch.setenv("HEDGE_WEBHOOK_QUEUE_FILE", str(tmp_path / "alerts.sqlite3"))
    with q._connect() as db:
        db.execute(
            """INSERT INTO hedge_alerts
            (received_at,bot,action,side,symbol,tp1_pct,dry_run,status,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (1.0, "Blaupunkt", "flip", "buy", "XBT-USDT", 0.5, 0, "interrupted", 1.0),
        )
    assert q._process_next() is False
    assert q.status()["counts"] == {"interrupted": 1}

def test_hedged_exchange_margin_netting_cannot_overallocate_virtual_pool(fleet):
    alert(fleet, "Blaupunkt", "flip", "buy")
    alert(fleet, "ElliotP1", "flip", "sell")
    assert fleet.broker.positions == {"long": 45, "short": 45}
    # Simulate KuCoin netting 45+45 opposing hedge margin and reporting
    # ample free margin. Per-bot virtual exposure still consumes 90% pool.
    fleet.broker.get_account_balance = lambda currency="USDT": {
        "equity": 1000, "available": 1000
    }
    assert alert(fleet, "ThirdBot", "exit")["reason"] == "already_flat"
    result = alert(fleet, "ThirdBot", "flip", "buy")
    assert result["ok"] is False
    assert result["reason"] == "insufficient_allocated_margin"
    assert fleet.broker.positions == {"long": 45, "short": 45}
