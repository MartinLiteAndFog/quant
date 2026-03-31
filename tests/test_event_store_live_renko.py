from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from quant.execution.event_store import (
    load_live_renko_bricks_from_postgres,
    prune_live_renko_bricks_before,
    upsert_live_renko_bricks,
)


class _Ctx:
    def __init__(self, value) -> None:
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeCursor:
    def __init__(self, rows=None, rowcount: int = 0) -> None:
        self.rows = list(rows or [])
        self.rowcount = int(rowcount)
        self.execute_calls = []
        self.executemany_calls = []

    def execute(self, sql, params=None) -> None:
        self.execute_calls.append((sql, params))

    def executemany(self, sql, rows) -> None:
        self.executemany_calls.append((sql, list(rows)))

    def fetchall(self):
        return list(self.rows)


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return _Ctx(self._cursor)


class LiveRenkoEventStoreTests(unittest.TestCase):
    def test_upsert_live_renko_bricks_writes_each_bar(self) -> None:
        cursor = _FakeCursor()
        conn = _FakeConn(cursor)
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-03-01T00:00:00Z", "2026-03-01T00:01:00Z"], utc=True),
                "open": [100.0, 100.1],
                "high": [100.2, 100.3],
                "low": [99.9, 100.0],
                "close": [100.1, 100.2],
            }
        )

        with patch("quant.execution.event_store.get_conn", return_value=_Ctx(conn)):
            written = upsert_live_renko_bricks(symbol="SOL-USDT", renko=df, source="test")

        self.assertEqual(written, 2)
        self.assertEqual(len(cursor.executemany_calls), 1)
        _, rows = cursor.executemany_calls[0]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["symbol"], "SOL-USDT")

    def test_load_live_renko_bricks_from_postgres_returns_sorted_frame(self) -> None:
        cursor = _FakeCursor(
            rows=[
                (pd.Timestamp("2026-03-01T00:01:00Z"), 100.1, 100.3, 100.0, 100.2),
                (pd.Timestamp("2026-03-01T00:00:00Z"), 100.0, 100.2, 99.9, 100.1),
            ]
        )
        conn = _FakeConn(cursor)

        with patch("quant.execution.event_store.get_conn", return_value=_Ctx(conn)):
            out = load_live_renko_bricks_from_postgres(symbol="SOL-USDT")

        self.assertEqual(len(out), 2)
        self.assertEqual(list(out.columns), ["ts", "open", "high", "low", "close"])
        self.assertTrue(out["ts"].is_monotonic_increasing)

    def test_prune_live_renko_bricks_before_returns_deleted_count(self) -> None:
        cursor = _FakeCursor(rowcount=7)
        conn = _FakeConn(cursor)

        with patch("quant.execution.event_store.get_conn", return_value=_Ctx(conn)):
            deleted = prune_live_renko_bricks_before(symbol="SOL-USDT", cutoff_ts="2026-03-01T00:00:00Z")

        self.assertEqual(deleted, 7)


if __name__ == "__main__":
    unittest.main()
