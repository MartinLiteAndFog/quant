from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from quant.brain_forward.evidence import ForwardProtocol
from quant.brain_forward import export


def _protocol() -> ForwardProtocol:
    return ForwardProtocol.from_dict({
        "schema_version": 1, "protocol_id": "export-test", "candidate_id": "candidate",
        "symbol": "SOL-USDT", "source": "binance_spot_klines",
        "artifact_sha256": "a" * 64, "candidate_spec_sha256": "b" * 64,
        "observer_code_sha256": {"runtime.py": "c" * 64, "service.py": "d" * 64, "store.py": "e" * 64},
        "warmup_start": "2026-01-01T00:00:00Z", "evidence_start": "2026-01-02T00:00:00Z",
        "evidence_end": "2026-01-02T01:00:00Z", "checkpoint_at": "2026-01-02T01:05:00Z",
        "outcome_maturity_minutes": 5,
        "base_cost_bps": 14, "stress_cost_bps": 22, "minimum_formal_trades": 2,
        "minimum_bar_coverage": 1.0, "maximum_drawdown_bps": 1000,
        "minimum_mean_net_bps": 0, "minimum_lcb95_net_bps": 0,
        "minimum_stress_mean_net_bps": 0, "promotion_scope": "shadow_champion_review_only",
        "live_orders_permitted": False,
    })


def _decision(protocol: ForwardProtocol, minute: int) -> dict[str, object]:
    event = protocol.evidence_start + timedelta(minutes=minute)
    return {
        "decision_id": f"decision-{minute}", "event_ts": event,
        "symbol": protocol.symbol, "source": protocol.source,
        "expected_net_bps": 8.0, "candle_range": 1.0, "active_memories": 1,
        "payload_json": {}, "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256, "artifact_sha256": protocol.artifact_sha256,
        "evidence_phase": "formal",
    }


def _trade(protocol: ForwardProtocol, minute: int) -> dict[str, object]:
    event = protocol.evidence_start + timedelta(minutes=minute)
    return {
        "decision_id": f"decision-{minute}", "event_ts": event,
        "entry_ts": event + timedelta(minutes=1), "exit_ts": event + timedelta(minutes=2),
        "entry_price": 100.0, "exit_price": 101.0, "target_price": 101.0, "stop_price": 99.0,
        "exit_reason": "target", "gross_bps": 60.0, "net_bps": 46.0, "expected_net_bps": 8.0,
        "protocol_id": protocol.protocol_id, "protocol_sha256": protocol.protocol_sha256,
        "artifact_sha256": protocol.artifact_sha256, "symbol": protocol.symbol,
        "source": protocol.source, "evidence_phase": "formal",
    }


class _Cursor:
    def __init__(self, protocol: ForwardProtocol) -> None:
        self.protocol = protocol
        self.executed: list[tuple[object, object]] = []
        self._stage = 0
        self.description: list[tuple[str]] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: object, params: object = None) -> None:
        self.executed.append((sql, params))
        self._stage += 1
        columns = {
            2: ["decision_id", "event_ts", "symbol", "source", "expected_net_bps", "candle_range", "active_memories", "payload_json", "protocol_id", "protocol_sha256", "artifact_sha256", "evidence_phase"],
            3: ["decision_id", "event_ts", "entry_ts", "exit_ts", "entry_price", "exit_price", "target_price", "stop_price", "exit_reason", "gross_bps", "net_bps", "expected_net_bps", "protocol_id", "protocol_sha256", "artifact_sha256", "symbol", "source", "evidence_phase"],
            4: ["ts"],
        }
        self.description = [(name,) for name in columns.get(self._stage, [])]

    def fetchone(self) -> tuple[str]:
        return (self.protocol.protocol_sha256,)

    def fetchall(self) -> list[tuple[object, ...]]:
        if self._stage == 2:
            return [tuple(_decision(self.protocol, minute)[name] for name, in self.description) for minute in (0, 4)]
        if self._stage == 3:
            return [tuple(_trade(self.protocol, minute)[name] for name, in self.description) for minute in (0, 4)]
        if self._stage == 4:
            return [(self.protocol.evidence_start + timedelta(minutes=minute),) for minute in range(65)]
        return []


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self.cursor_value = cursor

    def cursor(self) -> _Cursor:
        return self.cursor_value


def _connection(cursor: _Cursor):
    @contextmanager
    def factory():
        yield _Connection(cursor)
    return factory


class BrainForwardExportTests(unittest.TestCase):
    def test_protocol_scoped_export_queries_only_frozen_identity(self) -> None:
        protocol = _protocol()
        cursor = _Cursor(protocol)
        with patch.object(export, "_get_conn", _connection(cursor)):
            ledger = export.export_protocol_ledger(protocol)
        self.assertEqual(len(ledger["decisions"]), 2)
        self.assertEqual(len(ledger["trades"]), 2)
        self.assertEqual(len(ledger["bar_timestamps"]), 65)
        self.assertEqual(cursor.executed[1][1], (protocol.protocol_id, protocol.protocol_sha256))
        self.assertEqual(cursor.executed[2][1], (protocol.protocol_id, protocol.protocol_sha256))

    def test_bundle_is_hashed_and_never_overwrites_existing_evidence(self) -> None:
        protocol = _protocol()
        ledger = {
            "decisions": [_decision(protocol, 0), _decision(protocol, 4)],
            "trades": [_trade(protocol, 0), _trade(protocol, 4)],
            "bar_timestamps": [{"ts": protocol.evidence_start + timedelta(minutes=minute)} for minute in range(65)],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "evidence"
            report = export.write_evidence_bundle(protocol, ledger, as_of=protocol.checkpoint_at, output_dir=output_dir)
            self.assertEqual(report["verdict"], "eligible_for_shadow_champion_review")
            self.assertTrue((output_dir / "manifest.json").exists())
            with self.assertRaises(FileExistsError):
                export.write_evidence_bundle(protocol, ledger, as_of=protocol.checkpoint_at, output_dir=output_dir)


if __name__ == "__main__":
    unittest.main()
