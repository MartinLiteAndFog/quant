from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def default_regime_db_path() -> str:
    return os.getenv("REGIME_DB_PATH", "data/live/regime.db")


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class RegimeStateRecord:
    ts: str
    symbol: str
    gate_on: int
    regime_state: str
    regime_score: Optional[float] = None
    confidence: Optional[float] = None
    reason_code: Optional[str] = None
    model_version: Optional[str] = None
    threshold_snapshot_id: Optional[str] = None
    feature_values_json: Optional[str] = None


class RegimeStore:
    """
    SQLite-backed regime persistence with a portable schema.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = str(db_path or default_regime_db_path())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def init_schema(self) -> None:
        with self._conn() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS regime_state_ts (
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    gate_on INTEGER NOT NULL,
                    regime_state TEXT NOT NULL,
                    regime_score REAL,
                    confidence REAL,
                    reason_code TEXT,
                    model_version TEXT,
                    threshold_snapshot_id TEXT,
                    feature_values_json TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(symbol, ts)
                );

                CREATE TABLE IF NOT EXISTS regime_threshold_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    fitted_at TEXT NOT NULL,
                    train_window_start TEXT,
                    train_window_end TEXT,
                    dead_days INTEGER,
                    params_json TEXT,
                    model_version TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS regime_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    prev_state TEXT,
                    new_state TEXT NOT NULL,
                    trigger TEXT,
                    confirmation_bars INTEGER,
                    hold_time_prev_state_hours REAL,
                    debounce_applied INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS regime_data_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    missing_bars INTEGER,
                    stale_age_sec REAL,
                    source_name TEXT,
                    fallback_used INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_regime_state_symbol_ts
                    ON regime_state_ts(symbol, ts);
                CREATE INDEX IF NOT EXISTS idx_regime_transitions_symbol_ts
                    ON regime_transitions(symbol, ts);
                CREATE INDEX IF NOT EXISTS idx_regime_quality_symbol_ts
                    ON regime_data_quality(symbol, ts);
                """
            )

    def upsert_regime_state(self, rec: RegimeStateRecord) -> None:
        created_at = _utc_now_iso()
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO regime_state_ts (
                    ts, symbol, gate_on, regime_state, regime_score, confidence,
                    reason_code, model_version, threshold_snapshot_id, feature_values_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, ts) DO UPDATE SET
                    gate_on=excluded.gate_on,
                    regime_state=excluded.regime_state,
                    regime_score=excluded.regime_score,
                    confidence=excluded.confidence,
                    reason_code=excluded.reason_code,
                    model_version=excluded.model_version,
                    threshold_snapshot_id=excluded.threshold_snapshot_id,
                    feature_values_json=excluded.feature_values_json
                """,
                (
                    rec.ts,
                    rec.symbol,
                    int(rec.gate_on),
                    str(rec.regime_state),
                    rec.regime_score,
                    rec.confidence,
                    rec.reason_code,
                    rec.model_version,
                    rec.threshold_snapshot_id,
                    rec.feature_values_json,
                    created_at,
                ),
            )

    def insert_threshold_snapshot(
        self,
        *,
        snapshot_id: str,
        symbol: str,
        fitted_at: str,
        train_window_start: Optional[str] = None,
        train_window_end: Optional[str] = None,
        dead_days: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        model_version: Optional[str] = None,
    ) -> None:
        created_at = _utc_now_iso()
        params_json = json.dumps(params or {}, separators=(",", ":"), ensure_ascii=False)
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO regime_threshold_snapshots (
                    snapshot_id, symbol, fitted_at, train_window_start, train_window_end,
                    dead_days, params_json, model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    symbol=excluded.symbol,
                    fitted_at=excluded.fitted_at,
                    train_window_start=excluded.train_window_start,
                    train_window_end=excluded.train_window_end,
                    dead_days=excluded.dead_days,
                    params_json=excluded.params_json,
                    model_version=excluded.model_version
                """,
                (
                    snapshot_id,
                    symbol,
                    fitted_at,
                    train_window_start,
                    train_window_end,
                    dead_days,
                    params_json,
                    model_version,
                    created_at,
                ),
            )

    def insert_transition(
        self,
        *,
        ts: str,
        symbol: str,
        prev_state: Optional[str],
        new_state: str,
        trigger: Optional[str] = None,
        confirmation_bars: Optional[int] = None,
        hold_time_prev_state_hours: Optional[float] = None,
        debounce_applied: bool = False,
    ) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO regime_transitions (
                    ts, symbol, prev_state, new_state, trigger,
                    confirmation_bars, hold_time_prev_state_hours, debounce_applied, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    symbol,
                    prev_state,
                    new_state,
                    trigger,
                    confirmation_bars,
                    hold_time_prev_state_hours,
                    int(bool(debounce_applied)),
                    _utc_now_iso(),
                ),
            )

    def insert_data_quality(
        self,
        *,
        ts: str,
        symbol: str,
        missing_bars: Optional[int],
        stale_age_sec: Optional[float],
        source_name: Optional[str],
        fallback_used: bool = False,
    ) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO regime_data_quality (
                    ts, symbol, missing_bars, stale_age_sec, source_name, fallback_used, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    symbol,
                    missing_bars,
                    stale_age_sec,
                    source_name,
                    int(bool(fallback_used)),
                    _utc_now_iso(),
                ),
            )

    def get_latest_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            row = con.execute(
                """
                SELECT *
                FROM regime_state_ts
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_history(self, symbol: str, start_ts: Optional[str], end_ts: Optional[str], limit: int = 5000) -> List[Dict[str, Any]]:
        conds = ["symbol = ?"]
        args: List[Any] = [symbol]
        if start_ts:
            conds.append("ts >= ?")
            args.append(start_ts)
        if end_ts:
            conds.append("ts <= ?")
            args.append(end_ts)
        args.append(int(max(1, limit)))
        sql = f"""
            SELECT *
            FROM regime_state_ts
            WHERE {' AND '.join(conds)}
            ORDER BY ts ASC
            LIMIT ?
        """
        with self._conn() as con:
            rows = con.execute(sql, tuple(args)).fetchall()
        return [dict(r) for r in rows]

    def get_recent_transitions(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT *
                FROM regime_transitions
                WHERE symbol = ?
                ORDER BY ts DESC, id DESC
                LIMIT ?
                """,
                (symbol, int(max(1, limit))),
            ).fetchall()
        return [dict(r) for r in rows]
