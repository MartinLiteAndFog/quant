from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .store import RegimeStateRecord, RegimeStore


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _to_iso_utc(ts_like: Any) -> str:
    ts = pd.to_datetime(ts_like, utc=True, errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.utcnow().tz_localize("UTC")
    return ts.isoformat()


@dataclass
class RegimeDecision:
    ts: str
    symbol: str
    gate_on: int
    regime_state: str
    regime_score: float
    confidence: float
    reason_code: str
    model_version: str = "regime-v1"
    threshold_snapshot_id: Optional[str] = None
    feature_values: Optional[Dict[str, Any]] = None


def default_regime_state_for_gate(gate_on: int, on_state: str = "trend", off_state: str = "countertrend") -> str:
    return on_state if int(gate_on) else off_state


def compute_confidence(gate_on: int, regime_score: Optional[float], feature_values: Optional[Dict[str, Any]] = None) -> float:
    if feature_values and "confidence" in feature_values:
        try:
            return _clip(float(feature_values["confidence"]), 0.0, 1.0)
        except Exception:
            pass
    if regime_score is None:
        return 0.7 if int(gate_on) else 0.6
    # Deterministic confidence baseline from score magnitude.
    return _clip(0.35 + 0.65 * abs(float(regime_score)), 0.0, 1.0)


class RegimeService:
    def __init__(self, store: RegimeStore) -> None:
        self.store = store

    def upsert_decision(self, decision: RegimeDecision) -> Dict[str, Any]:
        ts_iso = _to_iso_utc(decision.ts)
        latest = self.store.get_latest_state(decision.symbol)
        prev_state = latest["regime_state"] if latest else None
        prev_ts = pd.to_datetime(latest["ts"], utc=True) if latest else None
        is_transition = prev_state is not None and prev_state != decision.regime_state

        rec = RegimeStateRecord(
            ts=ts_iso,
            symbol=decision.symbol,
            gate_on=int(decision.gate_on),
            regime_state=str(decision.regime_state),
            regime_score=float(decision.regime_score),
            confidence=float(decision.confidence),
            reason_code=str(decision.reason_code),
            model_version=str(decision.model_version),
            threshold_snapshot_id=decision.threshold_snapshot_id,
            feature_values_json=json.dumps(decision.feature_values or {}, separators=(",", ":"), ensure_ascii=False),
        )
        self.store.upsert_regime_state(rec)

        if is_transition:
            hold_hours = None
            if prev_ts is not None and pd.notna(prev_ts):
                hold_hours = float((pd.to_datetime(ts_iso, utc=True) - prev_ts).total_seconds() / 3600.0)
            self.store.insert_transition(
                ts=ts_iso,
                symbol=decision.symbol,
                prev_state=prev_state,
                new_state=decision.regime_state,
                trigger=decision.reason_code,
                confirmation_bars=None,
                hold_time_prev_state_hours=hold_hours,
                debounce_applied=False,
            )
        return {"ts": ts_iso, "symbol": decision.symbol, "transition": bool(is_transition), "prev_state": prev_state}

    def ingest_gate_dataframe(
        self,
        *,
        df: pd.DataFrame,
        symbol: str,
        gate_col: str = "gate_on",
        ts_col: str = "ts",
        score_col: Optional[str] = None,
        confidence_col: Optional[str] = None,
        reason_code: str = "ingest_gate_df",
        model_version: str = "regime-v1",
        threshold_snapshot_id: Optional[str] = None,
        on_state: str = "trend",
        off_state: str = "countertrend",
    ) -> int:
        if ts_col not in df.columns:
            raise ValueError(f"missing ts_col='{ts_col}'")
        if gate_col not in df.columns:
            raise ValueError(f"missing gate_col='{gate_col}'")

        work = df.copy()
        work[ts_col] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
        work[gate_col] = pd.to_numeric(work[gate_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        work = work.dropna(subset=[ts_col]).sort_values(ts_col).drop_duplicates(ts_col, keep="last")

        count = 0
        for _, row in work.iterrows():
            gate_on = int(row[gate_col])
            score_val = None
            if score_col and score_col in work.columns:
                try:
                    score_val = float(row[score_col])
                except Exception:
                    score_val = None
            if score_val is None:
                score_val = 1.0 if gate_on else -1.0

            conf_val = None
            if confidence_col and confidence_col in work.columns:
                try:
                    conf_val = float(row[confidence_col])
                except Exception:
                    conf_val = None
            if conf_val is None:
                conf_val = compute_confidence(gate_on, score_val, feature_values=None)

            d = RegimeDecision(
                ts=row[ts_col],
                symbol=symbol,
                gate_on=gate_on,
                regime_state=default_regime_state_for_gate(gate_on, on_state=on_state, off_state=off_state),
                regime_score=score_val,
                confidence=conf_val,
                reason_code=reason_code,
                model_version=model_version,
                threshold_snapshot_id=threshold_snapshot_id,
                feature_values=None,
            )
            self.upsert_decision(d)
            count += 1
        return count
