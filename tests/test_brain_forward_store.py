from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from quant.brain_forward.evidence import ForwardProtocol
from quant.brain_forward import store


def _protocol() -> ForwardProtocol:
    return ForwardProtocol.from_dict({
        "schema_version": 1, "protocol_id": "store-test", "candidate_id": "candidate",
        "symbol": "SOL-USDT", "source": "binance_spot_klines",
        "artifact_sha256": "a" * 64, "candidate_spec_sha256": "b" * 64,
        "observer_code_sha256": {"runtime.py": "c" * 64, "service.py": "d" * 64, "store.py": "e" * 64},
        "warmup_start": "2026-01-01T00:00:00Z", "evidence_start": "2026-01-02T00:00:00Z",
        "evidence_end": "2026-01-03T00:00:00Z", "checkpoint_at": "2026-01-03T00:05:00Z",
        "outcome_maturity_minutes": 5,
        "base_cost_bps": 14, "stress_cost_bps": 22, "minimum_formal_trades": 2,
        "minimum_bar_coverage": 0.99, "maximum_drawdown_bps": 1000,
        "minimum_mean_net_bps": 0, "minimum_lcb95_net_bps": 0,
        "minimum_stress_mean_net_bps": 0, "promotion_scope": "shadow_champion_review_only",
        "live_orders_permitted": False,
    })


class _Cursor:
    def __init__(self, fetched_hash: str | None = None) -> None:
        self.fetched_hash = fetched_hash
        self.executed: list[tuple[object, object]] = []
        self.many: list[tuple[object, list[dict[str, object]]]] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: object, params: object = None) -> None:
        self.executed.append((sql, params))

    def executemany(self, sql: object, rows: list[dict[str, object]]) -> None:
        self.many.append((sql, rows))

    def fetchone(self) -> tuple[str] | None:
        return (self.fetched_hash,) if self.fetched_hash is not None else None


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _Cursor:
        return self._cursor


def _connection_factory(cursor: _Cursor):
    @contextmanager
    def factory():
        yield _Connection(cursor)

    return factory


class BrainForwardStoreTests(unittest.TestCase):
    def test_schema_adds_immutable_protocol_identity_columns(self) -> None:
        cursor = _Cursor()
        with patch.object(store, "_get_conn", _connection_factory(cursor)):
            store.ensure_schema()
        sql = str(cursor.executed[0][0])
        self.assertIn("brain_forward_protocols", sql)
        self.assertIn("add column if not exists protocol_sha256", sql)
        self.assertIn("brain_forward_decisions add column if not exists artifact_sha256", sql)
        self.assertIn("add column if not exists evidence_phase", sql)

    def test_protocol_registration_rejects_same_id_with_different_hash(self) -> None:
        protocol = _protocol()
        cursor = _Cursor("0" * 64)
        with patch.object(store, "ensure_schema"), \
             patch.object(store, "_get_conn", _connection_factory(cursor)), \
             self.assertRaisesRegex(RuntimeError, "differs"):
            store.register_protocol(protocol)

    def test_decision_and_trade_rows_are_stamped_with_frozen_identity(self) -> None:
        protocol = _protocol()
        event_ts = pd.Timestamp("2026-01-02T01:00:00Z")
        decision = SimpleNamespace(
            event_ts=event_ts, expected_net_bps=5.0, candle_range=1.0,
            active_memories=2, shock_z=3.2, close_position=0.1,
            volatility_ratio=2.2, flow_imbalance=-0.2,
        )
        decision_cursor = _Cursor()
        with patch.object(store, "ensure_schema"), \
             patch.object(store, "_get_conn", _connection_factory(decision_cursor)):
            self.assertEqual(store.upsert_decisions(
                "SOL-USDT", "binance_spot_klines", [decision],
                artifact_sha256="a" * 64, protocol=protocol,
            ), 1)
        decision_row = decision_cursor.many[0][1][0]
        self.assertEqual(decision_row["protocol_id"], protocol.protocol_id)
        self.assertEqual(decision_row["protocol_sha256"], protocol.protocol_sha256)
        self.assertEqual(decision_row["artifact_sha256"], "a" * 64)
        self.assertEqual(decision_row["evidence_phase"], "formal")
        self.assertTrue(str(decision_row["decision_id"]).startswith("store-test:"))

        trade_cursor = _Cursor()
        trade = {
            "decision_id": "brain-forward:event", "event_ts": event_ts,
            "entry_ts": event_ts + pd.Timedelta(minutes=1),
            "exit_ts": event_ts + pd.Timedelta(minutes=2),
            "entry_price": 100.0, "exit_price": 101.0, "target_price": 101.0,
            "stop_price": 99.0, "exit_reason": "target", "gross_bps": 20.0,
            "net_bps": 6.0, "expected_net_bps": 5.0,
        }
        with patch.object(store, "ensure_schema"), \
             patch.object(store, "_get_conn", _connection_factory(trade_cursor)):
            self.assertEqual(store.upsert_trades(
                "SOL-USDT", "binance_spot_klines", [trade],
                artifact_sha256="a" * 64, protocol=protocol,
            ), 1)
        trade_row = trade_cursor.many[0][1][0]
        self.assertEqual(trade_row["artifact_sha256"], "a" * 64)
        self.assertEqual(trade_row["protocol_sha256"], protocol.protocol_sha256)
        self.assertEqual(trade_row["evidence_phase"], "formal")
        self.assertEqual(trade_row["decision_id"], "store-test:brain-forward:event")


if __name__ == "__main__":
    unittest.main()
