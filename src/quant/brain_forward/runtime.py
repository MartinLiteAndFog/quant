"""Exact paper-only signal runtime for the frozen five-minute utility memories."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ARTIFACT = Path(__file__).with_name("frozen_utility_memory.json")
BAR_SECONDS = 60
VOLATILITY_LOOKBACK = 1440
VOLATILITY_MIN_PERIODS = 240
SHORT_LOOKBACK = 60
SHORT_MIN_PERIODS = 30
MIN_SUPPORT = 15
COST_BPS = 14.0


@dataclass(frozen=True)
class BrainDecision:
    event_ts: pd.Timestamp
    event_close: float
    candle_range: float
    expected_net_bps: float
    active_memories: int
    shock_z: float
    close_position: float
    volatility_ratio: float
    flow_imbalance: float


def _digitize(values: np.ndarray, boundaries: tuple[float, ...]) -> np.ndarray:
    state = np.digitize(values, np.asarray(boundaries, dtype=float), right=False).astype(np.int64)
    state[~np.isfinite(values)] = -1
    return state


def _state_micro_flow(row: pd.DataFrame) -> np.ndarray:
    shock = _digitize(row["shock_z"].to_numpy(dtype=float), (3.0, 3.5, 4.0, 5.0, 6.0))
    flow = _digitize(row["flow_imbalance"].to_numpy(dtype=float), (-0.50, -0.30, -0.10, 0.10, 0.30, 0.50))
    return np.where((shock >= 0) & (flow >= 0), shock * 7 + flow, -1)


def _state_path(row: pd.DataFrame) -> np.ndarray:
    shock = _digitize(row["shock_z"].to_numpy(dtype=float), (3.0, 3.5, 4.0, 5.0, 6.0))
    ret5 = _digitize(row["ret5_z"].to_numpy(dtype=float), (-1.0, -0.25, 0.25, 1.0))
    ret30 = _digitize(row["ret30_z"].to_numpy(dtype=float), (-1.0, -0.25, 0.25, 1.0))
    return np.where((shock >= 0) & (ret5 >= 0) & (ret30 >= 0), (shock * 5 + ret5) * 5 + ret30, -1)


def _state_regime(row: pd.DataFrame) -> np.ndarray:
    shock = _digitize(row["shock_z"].to_numpy(dtype=float), (3.0, 3.5, 4.0, 5.0, 6.0))
    volatility = _digitize(row["volatility_ratio"].to_numpy(dtype=float), (0.75, 1.0, 1.5, 2.0))
    return np.where((shock >= 0) & (volatility >= 0), shock * 5 + volatility, -1)


STATE_FUNCTIONS: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "micro_flow": _state_micro_flow,
    "path": _state_path,
    "regime": _state_regime,
}


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the causal research features from closed one-minute bars."""

    required = {"ts", "open", "high", "low", "close", "volume", "taker_base"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"missing minute-bar columns: {missing}")
    frame = raw.loc[:, sorted(required)].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    for column in required.difference({"ts"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    if frame.empty:
        return frame
    frame["segment"] = (~frame["ts"].diff().eq(pd.Timedelta(seconds=BAR_SECONDS))).cumsum().astype(np.int64)
    segment = frame["segment"]
    log_close = np.log(frame["close"])
    r1 = log_close.groupby(segment, sort=False).diff()
    prior_r1 = r1.groupby(segment, sort=False).shift(1)
    vol_long = (
        prior_r1.groupby(segment, sort=False).rolling(VOLATILITY_LOOKBACK, min_periods=VOLATILITY_MIN_PERIODS)
        .std(ddof=0).reset_index(level=0, drop=True).reindex(frame.index)
    )
    vol_short = (
        prior_r1.groupby(segment, sort=False).rolling(SHORT_LOOKBACK, min_periods=SHORT_MIN_PERIODS)
        .std(ddof=0).reset_index(level=0, drop=True).reindex(frame.index)
    )
    ret5 = log_close - log_close.groupby(segment, sort=False).shift(5)
    ret30 = log_close - log_close.groupby(segment, sort=False).shift(30)
    candle_range = frame["high"] - frame["low"]
    scale = vol_long.replace(0.0, np.nan)
    output = frame.copy()
    output["r1"] = r1
    output["shock_z"] = r1.abs() / scale
    output["close_position"] = ((frame["close"] - frame["low"]) / candle_range.replace(0.0, np.nan)).clip(0.0, 1.0)
    output["candle_range"] = candle_range
    output["ret5_z"] = ret5 / (scale * np.sqrt(5.0))
    output["ret30_z"] = ret30 / (scale * np.sqrt(30.0))
    output["volatility_ratio"] = vol_short / scale
    output["flow_imbalance"] = (2.0 * (frame["taker_base"] / frame["volume"].replace(0.0, np.nan)).clip(0.0, 1.0)) - 1.0
    return output


class FrozenUtilityMemory:
    """Read-only utility ensemble generated from the frozen 31-Dec-2025 state."""

    def __init__(self, artifact: Path = ARTIFACT) -> None:
        # Railway's image retains the source tree even if an older setuptools
        # build omits package data.  Prefer the installed package, then use the
        # immutable source copy bundled into this dedicated observer image.
        artifact_path = artifact
        source_artifact = Path("/app/src/quant/brain_forward/frozen_utility_memory.json")
        if not artifact_path.exists() and source_artifact.exists():
            artifact_path = source_artifact
        artifact_bytes = artifact_path.read_bytes()
        data = json.loads(artifact_bytes)
        if data.get("schema_version") != 1 or data.get("model") != "cost_aware_thousand_brains_five_minute_v1":
            raise ValueError("unsupported frozen Brain artifact")
        self.metadata = data
        self.artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        self.prior = float(data["priors"]["1.0"])
        self.memory: dict[str, dict[int, tuple[int, float]]] = {}
        for name, rows in data["memories"].items():
            self.memory[name] = {
                int(row["state"]): (int(row["support"]), float(row["posterior_net_bps"]))
                for row in rows if float(row["barrier_multiplier"]) == 1.0
            }

    def predict(self, row: pd.DataFrame) -> tuple[float, int]:
        if len(row) != 1:
            raise ValueError("frozen utility memory predicts exactly one event row")
        evidence_sum = 0.0
        weight_sum = 0.0
        active_count = 0
        for name, state_function in STATE_FUNCTIONS.items():
            state = int(state_function(row)[0])
            support, posterior = self.memory[name].get(state, (0, self.prior))
            if support < MIN_SUPPORT:
                continue
            weight = min(np.sqrt(support / float(MIN_SUPPORT)), 3.0)
            evidence_sum += weight * (posterior - self.prior)
            weight_sum += weight
            active_count += 1
        return (self.prior + evidence_sum / weight_sum if weight_sum else self.prior), active_count

    def decide_latest(self, raw: pd.DataFrame) -> BrainDecision | None:
        features = build_feature_frame(raw)
        if features.empty:
            return None
        return self.decision_from_feature_row(features.tail(1).copy())

    def decision_from_feature_row(self, row: pd.DataFrame) -> BrainDecision | None:
        """Apply the frozen causal signal and utility gates to one feature row."""

        if len(row) != 1:
            raise ValueError("a Brain decision requires exactly one feature row")
        values = row.iloc[0]
        structural = bool(values["shock_z"] >= 3.0 and values["r1"] < 0.0 and values["close_position"] <= 0.20)
        bounds = bool(values["volatility_ratio"] >= 2.0 and -0.30 <= values["flow_imbalance"] < -0.10)
        if not structural or not bounds or not np.isfinite(values[["shock_z", "flow_imbalance", "volatility_ratio"]].to_numpy(dtype=float)).all():
            return None
        expected, active = self.predict(row)
        if expected < 0.0 or active < 1 or float(values["candle_range"]) <= 0.0:
            return None
        return BrainDecision(
            event_ts=pd.Timestamp(values["ts"]), event_close=float(values["close"]),
            candle_range=float(values["candle_range"]), expected_net_bps=float(expected), active_memories=active,
            shock_z=float(values["shock_z"]), close_position=float(values["close_position"]),
            volatility_ratio=float(values["volatility_ratio"]), flow_imbalance=float(values["flow_imbalance"]),
        )


def parse_binance_klines(rows: list[list[Any]], now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Parse only fully closed Binance one-minute klines with native taker-base volume."""

    current = pd.Timestamp.now("UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    parsed: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 10:
            continue
        try:
            close_ts = pd.to_datetime(int(row[6]), unit="ms", utc=True)
            if close_ts >= current:
                continue
            parsed.append({
                "ts": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "open": float(row[1]), "high": float(row[2]), "low": float(row[3]), "close": float(row[4]),
                "volume": float(row[5]), "taker_base": float(row[9]),
            })
        except (IndexError, TypeError, ValueError, OverflowError):
            continue
    return pd.DataFrame.from_records(parsed, columns=["ts", "open", "high", "low", "close", "volume", "taker_base"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)


def completed_paper_trades(raw: pd.DataFrame, model: FrozenUtilityMemory) -> tuple[list[BrainDecision], list[dict[str, object]]]:
    """Rebuild the causally selected, already-complete paper trades in a bar window."""

    # Use the feature builder's normalized frame as the execution frame as
    # well.  Maintaining two independently cleaned frames can shift numeric
    # row indices when an incomplete bar is dropped, which would attach a
    # valid signal to the wrong next-minute open.
    features = build_feature_frame(raw)
    bars = features.loc[
        :, ["ts", "open", "high", "low", "close", "segment"]
    ].copy()
    decisions: list[tuple[int, BrainDecision]] = []
    for index in range(len(features)):
        decision = model.decision_from_feature_row(features.iloc[[index]])
        if decision is not None:
            decisions.append((index, decision))
    selected = [decision for _, decision in decisions]
    rows: list[dict[str, object]] = []
    last_exit = -1
    for event_index, decision in decisions:
        entry_index = event_index + 1
        end_index = event_index + 5
        if end_index >= len(bars):
            continue
        if entry_index <= last_exit:
            continue
        # The research simulator rejects an event when any part of its
        # five-minute path crosses a missing-minute boundary.  Treating the
        # next observed bar after an outage as the next minute would fabricate
        # both holding time and fill reachability.
        if (
            int(bars.iloc[entry_index]["segment"])
            != int(bars.iloc[event_index]["segment"])
            or int(bars.iloc[end_index]["segment"])
            != int(bars.iloc[event_index]["segment"])
        ):
            continue
        entry = float(bars.iloc[entry_index]["open"])
        target = entry + decision.candle_range
        stop = entry - decision.candle_range
        exit_index, exit_price, reason = end_index, float(bars.iloc[end_index]["close"]), "time"
        for bar_index in range(entry_index, end_index + 1):
            bar = bars.iloc[bar_index]
            if float(bar["open"]) <= stop:
                exit_index, exit_price, reason = bar_index, float(bar["open"]), "stop_gap"
                break
            if float(bar["open"]) >= target:
                exit_index, exit_price, reason = bar_index, target, "target_gap"
                break
            up_hit = float(bar["high"]) >= target
            down_hit = float(bar["low"]) <= stop
            if down_hit:
                exit_index, exit_price, reason = bar_index, stop, "ambiguous_stop" if up_hit else "stop"
                break
            if up_hit:
                exit_index, exit_price, reason = bar_index, target, "target"
                break
        gross_bps = float(np.log(exit_price / entry) * 10_000.0)
        rows.append({
            "decision_id": f"brain-forward:{decision.event_ts.isoformat()}",
            "event_ts": decision.event_ts,
            "entry_ts": pd.Timestamp(bars.iloc[entry_index]["ts"]),
            "exit_ts": pd.Timestamp(bars.iloc[exit_index]["ts"]),
            "entry_price": entry, "exit_price": exit_price, "target_price": target, "stop_price": stop,
            "exit_reason": reason, "gross_bps": gross_bps, "net_bps": gross_bps - COST_BPS,
            "expected_net_bps": decision.expected_net_bps,
        })
        last_exit = exit_index
    return selected, rows
