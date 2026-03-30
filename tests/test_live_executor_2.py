from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from quant.execution.live_executor_2 import (
    ExecutorState,
    _arm_ttp_reenter_handoff,
    _derive_action_event_fields,
    _mark_ttp_reenter_attempt,
    _record_ttp_external_exit,
    _ttp_reenter_attempt_allowed,
    _ttp_reenter_handoff_action,
    run_once,
)


class _FlatBroker:
    def __init__(self) -> None:
        self.client = type("Client", (), {"cancel_all_reduce_only_orders": lambda *args, **kwargs: None})()

    def get_best_bid_ask(self, symbol: str):
        return 100.0, 100.0

    def get_position(self, symbol: str) -> float:
        return 0.0

    def get_account_balance(self, currency: str = "USDT"):
        return {"equity": 1000.0, "available": 1000.0, "margin": 0.0, "unrealised_pnl": 0.0}

    def get_contract_multiplier(self, symbol: str) -> float:
        return 1.0


class _FlatOms:
    def __init__(self) -> None:
        self.enter_market_calls: list[tuple[str, str, float]] = []
        self.arm_stop_entry_calls: list[tuple[str, str, float, float, str]] = []

    def find_stop_order_by_kind(self, symbol: str, kind: str):
        return None

    def cancel_orders_by_kind(self, symbol: str, kind: str) -> int:
        return 0

    def arm_stop_entry(self, symbol: str, side: str, qty: float, stop_price: float, kind: str):
        self.arm_stop_entry_calls.append((symbol, side, float(qty), float(stop_price), kind))
        return {"ok": True}

    def arm_stop_exit(self, *args, **kwargs):
        return {"ok": True}

    def arm_take_profit_exit(self, *args, **kwargs):
        return {"ok": True}

    def enter_market(self, symbol: str, side: str, qty: float):
        self.enter_market_calls.append((symbol, side, float(qty)))
        return {"ok": True, "order_id": "reenter-1", "client_id": "quant:test"}


class _LiveBroker:
    def __init__(self, pos: float, bid: float = 100.0, ask: float = 100.0) -> None:
        self._pos = float(pos)
        self._bid = float(bid)
        self._ask = float(ask)
        self.client = type("Client", (), {"cancel_all_reduce_only_orders": lambda *args, **kwargs: None})()

    def get_best_bid_ask(self, symbol: str):
        return self._bid, self._ask

    def get_position(self, symbol: str) -> float:
        return self._pos

    def get_account_balance(self, currency: str = "USDT"):
        return {"equity": 1000.0, "available": 1000.0, "margin": 0.0, "unrealised_pnl": 0.0}

    def get_contract_multiplier(self, symbol: str) -> float:
        return 1.0


class _LiveOms:
    def __init__(self) -> None:
        self.enter_market_calls: list[tuple[str, str, float]] = []
        self.flatten_market_calls: list[tuple[str, str, float]] = []
        self.stop_exit_calls: list[tuple[str, str, float, float, str]] = []
        self.cancel_calls: list[tuple[str, str]] = []

    def find_stop_order_by_kind(self, symbol: str, kind: str):
        return None

    def cancel_orders_by_kind(self, symbol: str, kind: str) -> int:
        self.cancel_calls.append((symbol, kind))
        return 0

    def arm_stop_exit(self, symbol: str, side: str, qty: float, stop_price: float, kind: str, reduce_only: bool = True):
        self.stop_exit_calls.append((symbol, side, float(qty), float(stop_price), kind))
        return {"ok": True}

    def arm_stop_entry(self, symbol: str, side: str, qty: float, stop_price: float, kind: str):
        return {"ok": True}

    def arm_take_profit_exit(self, *args, **kwargs):
        return {"ok": True}

    def enter_market(self, symbol: str, side: str, qty: float):
        self.enter_market_calls.append((symbol, side, float(qty)))
        return {"ok": True, "order_id": "entry-1", "client_id": "quant:test"}

    def flatten_market(self, symbol: str, side: str, qty: float):
        self.flatten_market_calls.append((symbol, side, float(qty)))
        return {"ok": True, "order_id": "flat-1", "client_id": "quant:test"}


class LiveExecutor2TtpTests(unittest.TestCase):
    def test_ttp_handoff_action_reuses_pending_context_when_already_flat(self) -> None:
        state = ExecutorState()
        _arm_ttp_reenter_handoff(
            state,
            prior_side="long",
            target_side="short",
            source_ts="2026-03-30T12:00:00Z",
        )

        action = _ttp_reenter_handoff_action(
            state,
            current_side="flat",
            mid=100.0,
            terminal={"mode": "TTP", "ttp": 99.0},
            source_ts="2026-03-30T12:00:01Z",
        )

        self.assertIsNotNone(action)
        self.assertEqual(action["action"], "ttp_confirm_reenter_short")
        self.assertEqual(action["prior_side"], "long")
        self.assertEqual(action["target_side"], "short")

    def test_derive_action_event_fields_special_cases_ttp_reenter(self) -> None:
        action_side, position_before, position_after = _derive_action_event_fields(
            action="ttp_confirm_reenter_short",
            current_side="long",
            want_side="long",
            terminal_pos=1,
            ttp_prior_side="long",
        )

        self.assertEqual(action_side, "short")
        self.assertEqual(position_before, 1)
        self.assertEqual(position_after, -1)

    def test_ttp_reenter_guard_blocks_same_handoff_until_cooldown_expires(self) -> None:
        state = ExecutorState()
        handoff_key = "long:short:2026-03-30T12:00:00Z"

        self.assertTrue(_ttp_reenter_attempt_allowed(state, handoff_key))

        _mark_ttp_reenter_attempt(state, handoff_key, cooldown_sec=10.0)
        self.assertFalse(_ttp_reenter_attempt_allowed(state, handoff_key))

        state.ttp_reenter_cooldown_until = "2000-01-01T00:00:00Z"
        self.assertTrue(_ttp_reenter_attempt_allowed(state, handoff_key))

    def test_record_ttp_external_exit_closes_old_leg_and_clears_open_leg_state(self) -> None:
        state = ExecutorState(
            open_leg_mode="flip",
            open_leg_id="leg-old",
            open_leg_side="long",
            open_leg_entry_bar_ts="2026-03-30T11:00:00Z",
        )
        calls: list[str] = []

        with (
            patch("quant.execution.live_executor_2._append_execution_event", side_effect=lambda **_: calls.append("execution")),
            patch("quant.execution.live_executor_2._append_closed_trade", side_effect=lambda **_: calls.append("closed_trade")),
        ):
            _record_ttp_external_exit(
                state,
                symbol="SOL-USDT",
                prior_side="long",
                terminal={
                    "entry_px": 100.0,
                    "entry_bar_ts": "2026-03-30T11:00:00Z",
                },
                ttp_px=99.0,
                event_name="ttp_external_exit",
                action="ttp_confirm_reenter_short",
                execution_seq=7,
                qty=3.0,
                exit_details={"price": 99.0, "qty": 3.0, "order_id": "ttp-exit"},
            )

        self.assertEqual(calls, ["execution", "closed_trade"])
        self.assertIsNone(state.open_leg_mode)
        self.assertIsNone(state.open_leg_id)
        self.assertIsNone(state.open_leg_side)
        self.assertIsNone(state.open_leg_entry_bar_ts)

    def test_run_once_reenters_from_flat_pending_ttp_handoff(self) -> None:
        state = ExecutorState(
            latched_exit_engine="flip",
            open_leg_mode="flip",
            open_leg_id="leg-old",
            open_leg_side="long",
            open_leg_entry_bar_ts="2026-03-30T11:00:00Z",
        )
        _arm_ttp_reenter_handoff(
            state,
            prior_side="long",
            target_side="short",
            source_ts="2026-03-30T12:00:00Z",
        )
        broker = _FlatBroker()
        oms = _FlatOms()
        events: list[str] = []

        with (
            patch("quant.execution.live_executor_2.get_live_gate_state", return_value={"gate_on": 1, "gate_countertrend_on": 1, "gate_trend_on": 0}),
            patch("quant.execution.live_executor_2._load_renko_bars", return_value=pd.DataFrame({"ts": pd.to_datetime(["2026-03-30T12:00:00Z"], utc=True), "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]})),
            patch("quant.execution.live_executor_2._load_signals_df", return_value=pd.DataFrame()),
            patch("quant.execution.live_executor_2._latest_backtest_event", return_value=({"ts": "2026-03-30T12:00:00Z", "event": "ttp_on", "side": 1, "seq": 1}, {"mode": "TTP", "pos": 0, "ttp": 99.0, "entry_px": 100.0, "entry_bar_ts": "2026-03-30T11:00:00Z"})),
            patch("quant.execution.live_executor_2.get_latest_imba_barriers", return_value={"ts": None, "long_barrier": 101.0, "short_barrier": 99.0}),
            patch("quant.execution.live_executor_2.write_execution_state", return_value={}),
            patch("quant.execution.live_executor_2._write_dashboard_levels", return_value=None),
            patch("quant.execution.live_executor_2._append_action_event", side_effect=lambda **_: events.append("action")),
            patch("quant.execution.live_executor_2._append_execution_event", side_effect=lambda **_: events.append("execution")),
            patch("quant.execution.live_executor_2._append_closed_trade", side_effect=lambda **_: events.append("closed_trade")),
            patch("quant.execution.live_executor_2._append_equity_snapshot", return_value=None),
            patch("quant.execution.live_executor_2._verify_execution_fill_ratio", return_value=None),
            patch("quant.execution.live_executor_2._sync_kraken_stop_loss", return_value=None),
            patch("quant.execution.live_executor_2.record_expected", return_value=None),
        ):
            state = run_once(
                broker=broker,
                oms=oms,
                symbol="SOL-USDT",
                signals_root=Path("unused"),
                state=state,
                live_enabled=True,
                dry_run=False,
                leverage=1.0,
            )

        self.assertEqual(state.last_action, "ttp_confirm_reenter_short")
        self.assertEqual(len(oms.enter_market_calls), 1)
        self.assertEqual(oms.enter_market_calls[0][1], "short")
        self.assertEqual(len(oms.arm_stop_entry_calls), 0)
        self.assertEqual(events.count("closed_trade"), 1)

    def test_wait_mode_does_not_flip_or_exit_from_terminal_picture(self) -> None:
        state = ExecutorState(latched_exit_engine="flip")
        broker = _LiveBroker(pos=5.0)
        oms = _LiveOms()

        with (
            patch("quant.execution.live_executor_2.get_live_gate_state", return_value={"gate_on": 1, "gate_countertrend_on": 1, "gate_trend_on": 0}),
            patch("quant.execution.live_executor_2._load_renko_bars", return_value=pd.DataFrame({"ts": pd.to_datetime(["2026-03-30T12:00:00Z"], utc=True), "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]})),
            patch("quant.execution.live_executor_2._load_signals_df", return_value=pd.DataFrame()),
            patch("quant.execution.live_executor_2._latest_backtest_event", return_value=({"ts": "2026-03-30T12:00:00Z", "event": "wait_mode", "side": -1, "seq": 1}, {"mode": "WAIT", "pos": -1, "side": "short", "sl": 95.0, "entry_px": 100.0, "entry_bar_ts": "2026-03-30T11:00:00Z"})),
            patch("quant.execution.live_executor_2.get_latest_imba_barriers", return_value={"ts": None, "long_barrier": 101.0, "short_barrier": 99.0}),
            patch("quant.execution.live_executor_2._latest_signal", return_value=None),
            patch("quant.execution.live_executor_2.write_execution_state", return_value={}),
            patch("quant.execution.live_executor_2._write_dashboard_levels", return_value=None),
            patch("quant.execution.live_executor_2._append_action_event", return_value=None),
            patch("quant.execution.live_executor_2._append_execution_event", return_value=None),
            patch("quant.execution.live_executor_2._append_equity_snapshot", return_value=None),
            patch("quant.execution.live_executor_2._verify_execution_fill_ratio", return_value=None),
            patch("quant.execution.live_executor_2._sync_kraken_stop_loss", return_value=None),
            patch("quant.execution.live_executor_2.record_expected", return_value=None),
        ):
            state = run_once(
                broker=broker,
                oms=oms,
                symbol="SOL-USDT",
                signals_root=Path("unused"),
                state=state,
                live_enabled=True,
                dry_run=False,
                leverage=1.0,
            )

        self.assertEqual(state.last_action, "hold")
        self.assertEqual(oms.flatten_market_calls, [])
        self.assertEqual(oms.enter_market_calls, [])

    def test_wait_same_side_imba_only_switches_order_set_to_ttp(self) -> None:
        state = ExecutorState(latched_exit_engine="flip")
        broker = _LiveBroker(pos=5.0)
        oms = _LiveOms()

        with (
            patch("quant.execution.live_executor_2.get_live_gate_state", return_value={"gate_on": 1, "gate_countertrend_on": 1, "gate_trend_on": 0}),
            patch("quant.execution.live_executor_2._load_renko_bars", return_value=pd.DataFrame({"ts": pd.to_datetime(["2026-03-30T12:00:00Z"], utc=True), "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]})),
            patch("quant.execution.live_executor_2._load_signals_df", return_value=pd.DataFrame()),
            patch("quant.execution.live_executor_2._latest_backtest_event", return_value=({"ts": "2026-03-30T12:00:00Z", "event": "wait_mode", "side": 1, "seq": 1}, {"mode": "WAIT", "pos": 1, "side": "long", "sl": 95.0, "ttp": 99.0, "entry_px": 100.0, "entry_bar_ts": "2026-03-30T11:00:00Z"})),
            patch("quant.execution.live_executor_2.get_latest_imba_barriers", return_value={"ts": None, "long_barrier": 101.0, "short_barrier": 99.0}),
            patch("quant.execution.live_executor_2._latest_signal", return_value={"ts": pd.Timestamp("2026-03-30T12:00:00Z"), "signal": 1}),
            patch("quant.execution.live_executor_2.write_execution_state", return_value={}),
            patch("quant.execution.live_executor_2._write_dashboard_levels", return_value=None),
            patch("quant.execution.live_executor_2._append_action_event", return_value=None),
            patch("quant.execution.live_executor_2._append_execution_event", return_value=None),
            patch("quant.execution.live_executor_2._append_equity_snapshot", return_value=None),
            patch("quant.execution.live_executor_2._verify_execution_fill_ratio", return_value=None),
            patch("quant.execution.live_executor_2._sync_kraken_stop_loss", return_value=None),
            patch("quant.execution.live_executor_2.record_expected", return_value=None),
        ):
            state = run_once(
                broker=broker,
                oms=oms,
                symbol="SOL-USDT",
                signals_root=Path("unused"),
                state=state,
                live_enabled=True,
                dry_run=False,
                leverage=1.0,
            )

        self.assertEqual(state.last_action, "sync_wait_to_ttp_orders")
        self.assertIn(("SOL-USDT", "long", 5.0, 99.0, "ttp_exit"), oms.stop_exit_calls)
        self.assertNotIn(("SOL-USDT", "long", 5.0, 95.0, "wait_sl"), oms.stop_exit_calls)
        self.assertEqual(oms.enter_market_calls, [])
        self.assertEqual(oms.flatten_market_calls, [])

    def test_pending_follow_entry_stays_armed_while_source_side_still_open(self) -> None:
        state = ExecutorState(
            latched_exit_engine="flip",
            pending_follow_entry=True,
            pending_follow_entry_side="short",
            pending_follow_entry_reason="opposite_imba_close",
            pending_follow_entry_source_ts="2026-03-30T12:00:00Z",
            pending_follow_entry_expires_at="2099-01-01T00:00:00Z",
        )
        state.pending_follow_entry_source_side = "long"  # type: ignore[attr-defined]
        broker = _LiveBroker(pos=5.0)
        oms = _LiveOms()

        with (
            patch("quant.execution.live_executor_2.get_live_gate_state", return_value={"gate_on": 1, "gate_countertrend_on": 1, "gate_trend_on": 0}),
            patch("quant.execution.live_executor_2._load_renko_bars", return_value=pd.DataFrame({"ts": pd.to_datetime(["2026-03-30T12:00:00Z"], utc=True), "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]})),
            patch("quant.execution.live_executor_2._load_signals_df", return_value=pd.DataFrame()),
            patch("quant.execution.live_executor_2._latest_backtest_event", return_value=({"ts": "2026-03-30T12:00:00Z", "event": "hold", "side": 1, "seq": 1}, {"mode": "TTP", "pos": 1, "side": "long", "ttp": 98.0, "entry_px": 100.0, "entry_bar_ts": "2026-03-30T11:00:00Z"})),
            patch("quant.execution.live_executor_2.get_latest_imba_barriers", return_value={"ts": None, "long_barrier": 101.0, "short_barrier": 99.0}),
            patch("quant.execution.live_executor_2._latest_signal", return_value=None),
            patch("quant.execution.live_executor_2.write_execution_state", return_value={}),
            patch("quant.execution.live_executor_2._write_dashboard_levels", return_value=None),
            patch("quant.execution.live_executor_2._append_action_event", return_value=None),
            patch("quant.execution.live_executor_2._append_execution_event", return_value=None),
            patch("quant.execution.live_executor_2._append_equity_snapshot", return_value=None),
            patch("quant.execution.live_executor_2._verify_execution_fill_ratio", return_value=None),
            patch("quant.execution.live_executor_2._sync_kraken_stop_loss", return_value=None),
            patch("quant.execution.live_executor_2.record_expected", return_value=None),
        ):
            state = run_once(
                broker=broker,
                oms=oms,
                symbol="SOL-USDT",
                signals_root=Path("unused"),
                state=state,
                live_enabled=True,
                dry_run=False,
                leverage=1.0,
            )

        self.assertTrue(state.pending_follow_entry)
        self.assertEqual(state.pending_follow_entry_side, "short")
        self.assertEqual(oms.enter_market_calls, [])

    def test_pending_follow_entry_reenters_without_waiting_for_terminal_confirmation(self) -> None:
        state = ExecutorState(
            latched_exit_engine="flip",
            pending_follow_entry=True,
            pending_follow_entry_side="short",
            pending_follow_entry_reason="opposite_imba_close",
            pending_follow_entry_source_ts="2026-03-30T12:00:00Z",
            pending_follow_entry_expires_at="2099-01-01T00:00:00Z",
        )
        state.pending_follow_entry_source_side = "long"  # type: ignore[attr-defined]
        broker = _FlatBroker()
        oms = _FlatOms()

        with (
            patch("quant.execution.live_executor_2.get_live_gate_state", return_value={"gate_on": 1, "gate_countertrend_on": 1, "gate_trend_on": 0}),
            patch("quant.execution.live_executor_2._load_renko_bars", return_value=pd.DataFrame({"ts": pd.to_datetime(["2026-03-30T12:00:00Z"], utc=True), "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]})),
            patch("quant.execution.live_executor_2._load_signals_df", return_value=pd.DataFrame()),
            patch("quant.execution.live_executor_2._latest_backtest_event", return_value=({"ts": "2026-03-30T12:00:00Z", "event": "flat_after_close", "side": 0, "seq": 1}, {"mode": "WAIT", "pos": 0, "side": None, "sl": 95.0, "entry_px": 100.0, "entry_bar_ts": "2026-03-30T11:00:00Z"})),
            patch("quant.execution.live_executor_2.get_latest_imba_barriers", return_value={"ts": None, "long_barrier": 101.0, "short_barrier": 99.0}),
            patch("quant.execution.live_executor_2._latest_signal", return_value=None),
            patch("quant.execution.live_executor_2.write_execution_state", return_value={}),
            patch("quant.execution.live_executor_2._write_dashboard_levels", return_value=None),
            patch("quant.execution.live_executor_2._append_action_event", return_value=None),
            patch("quant.execution.live_executor_2._append_execution_event", return_value=None),
            patch("quant.execution.live_executor_2._append_equity_snapshot", return_value=None),
            patch("quant.execution.live_executor_2._verify_execution_fill_ratio", return_value=None),
            patch("quant.execution.live_executor_2._sync_kraken_stop_loss", return_value=None),
            patch("quant.execution.live_executor_2.record_expected", return_value=None),
        ):
            state = run_once(
                broker=broker,
                oms=oms,
                symbol="SOL-USDT",
                signals_root=Path("unused"),
                state=state,
                live_enabled=True,
                dry_run=False,
                leverage=1.0,
            )

        self.assertFalse(state.pending_follow_entry)
        self.assertEqual(len(oms.enter_market_calls), 1)
        self.assertEqual(oms.enter_market_calls[0][1], "short")

    def test_opposite_imba_cross_arms_flip_process_without_direct_market_action(self) -> None:
        state = ExecutorState(latched_exit_engine="flip")
        broker = _LiveBroker(pos=9.0, bid=98.0, ask=98.0)
        oms = _LiveOms()

        with (
            patch("quant.execution.live_executor_2.get_live_gate_state", return_value={"gate_on": 1, "gate_countertrend_on": 1, "gate_trend_on": 0}),
            patch("quant.execution.live_executor_2._load_renko_bars", return_value=pd.DataFrame({"ts": pd.to_datetime(["2026-03-30T12:00:00Z"], utc=True), "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]})),
            patch("quant.execution.live_executor_2._load_signals_df", return_value=pd.DataFrame()),
            patch("quant.execution.live_executor_2._latest_backtest_event", return_value=({"ts": "2026-03-30T12:00:00Z", "event": "tp2_live", "side": 1, "seq": 1}, {"mode": "TP2", "pos": 1, "side": "long", "sl": 95.0, "tp1": 104.0, "tp2": 108.0, "entry_px": 100.0, "entry_bar_ts": "2026-03-30T11:00:00Z"})),
            patch("quant.execution.live_executor_2.get_latest_imba_barriers", return_value={"ts": None, "long_barrier": 101.0, "short_barrier": 99.0}),
            patch("quant.execution.live_executor_2._latest_signal", return_value={"ts": pd.Timestamp("2026-03-30T12:00:00Z"), "signal": -1}),
            patch("quant.execution.live_executor_2.write_execution_state", return_value={}),
            patch("quant.execution.live_executor_2._write_dashboard_levels", return_value=None),
            patch("quant.execution.live_executor_2._append_action_event", return_value=None),
            patch("quant.execution.live_executor_2._append_execution_event", return_value=None),
            patch("quant.execution.live_executor_2._append_equity_snapshot", return_value=None),
            patch("quant.execution.live_executor_2._verify_execution_fill_ratio", return_value=None),
            patch("quant.execution.live_executor_2._sync_kraken_stop_loss", return_value=None),
            patch("quant.execution.live_executor_2.record_expected", return_value=None),
        ):
            state = run_once(
                broker=broker,
                oms=oms,
                symbol="SOL-USDT",
                signals_root=Path("unused"),
                state=state,
                live_enabled=True,
                dry_run=False,
                leverage=1.0,
            )

        self.assertTrue(state.pending_follow_entry)
        self.assertEqual(state.pending_follow_entry_side, "short")
        self.assertEqual(oms.enter_market_calls, [])
        self.assertEqual(oms.flatten_market_calls, [])


if __name__ == "__main__":
    unittest.main()
