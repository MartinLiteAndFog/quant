from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd

from quant.brain_forward import variant_store
from quant.brain_forward.evidence import ForwardProtocol


def _protocol() -> ForwardProtocol:
    return ForwardProtocol.from_dict({
        "schema_version": 1, "protocol_id": "store-test", "candidate_id": "candidate",
        "symbol": "SOL-USDT", "source": "binance_spot_klines",
        "artifact_sha256": "a" * 64, "candidate_spec_sha256": "b" * 64,
        "observer_code_sha256": {
            "runtime.py": "c" * 64, "service.py": "d" * 64, "store.py": "e" * 64,
        },
        "warmup_start": "2026-01-01T00:00:00Z", "evidence_start": "2026-01-02T00:00:00Z",
        "evidence_end": "2026-01-03T00:00:00Z", "checkpoint_at": "2026-01-03T00:05:00Z",
        "outcome_maturity_minutes": 5, "base_cost_bps": 14, "stress_cost_bps": 22,
        "minimum_formal_trades": 2, "minimum_bar_coverage": 0.99,
        "maximum_drawdown_bps": 1000, "minimum_mean_net_bps": 0,
        "minimum_lcb95_net_bps": 0, "minimum_stress_mean_net_bps": 0,
        "promotion_scope": "shadow_champion_review_only", "live_orders_permitted": False,
    })


class _Cursor:
    def __init__(self) -> None:
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


class BrainForwardVariantStoreTests(unittest.TestCase):
    def test_schema_contains_separate_event_and_trade_ledgers(self) -> None:
        cursor = _Cursor()
        with patch.object(variant_store, "_get_conn", _connection_factory(cursor)):
            variant_store.ensure_variant_schema()
        sql = str(cursor.executed[0][0])
        self.assertIn("brain_forward_variant_events", sql)
        self.assertIn("brain_forward_variant_trades", sql)

    def test_event_upsert_stamps_protocol_identity_and_status(self) -> None:
        cursor = _Cursor()
        protocol = _protocol()
        event_ts = pd.Timestamp("2026-01-02T01:00:00Z")
        event = {
            "candidate_id": "brain-forward:immediate:event",
            "variant_id": "immediate",
            "event_ts": event_ts,
            "status": "triggered",
            "reason": "next_minute_open",
            "trigger_ts": event_ts + pd.Timedelta(minutes=1),
            "entry_ts": event_ts + pd.Timedelta(minutes=1),
            "entry_price": 100.0,
            "payload": {"expected_net_bps": 5.0},
        }
        with patch.object(variant_store, "ensure_variant_schema"), \
             patch.object(variant_store, "_get_conn", _connection_factory(cursor)):
            self.assertEqual(variant_store.upsert_variant_events(
                [event], artifact_sha256="a" * 64, protocol=protocol
            ), 1)
        row = cursor.many[0][1][0]
        self.assertEqual(row["protocol_id"], protocol.protocol_id)
        self.assertEqual(row["status"], "triggered")
        self.assertTrue(str(row["candidate_id"]).startswith("store-test:"))


if __name__ == "__main__":
    unittest.main()
