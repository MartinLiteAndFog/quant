"""Comparable paper-only entry variants for the frozen Brain signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant.brain_forward.runtime import (
    COST_BPS,
    BrainDecision,
    FrozenUtilityMemory,
    build_feature_frame,
)


IMMEDIATE = "immediate"
STOP_COOLDOWN_3M = "stop_cooldown_3m"
PREVIOUS_HIGH_CONFIRMATION = "previous_high_confirmation"
VARIANTS = (IMMEDIATE, STOP_COOLDOWN_3M, PREVIOUS_HIGH_CONFIRMATION)
HOLDING_BARS = 5
CONFIRMATION_BARS = 5
COOLDOWN_MINUTES = 3


@dataclass(frozen=True)
class VariantEvaluation:
    decisions: list[BrainDecision]
    events: list[dict[str, Any]]
    trades: list[dict[str, Any]]


def _event(
    variant_id: str,
    decision: BrainDecision,
    status: str,
    *,
    reason: str,
    trigger_ts: pd.Timestamp | None = None,
    entry_ts: pd.Timestamp | None = None,
    entry_price: float | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": f"brain-forward:{variant_id}:{decision.event_ts.isoformat()}",
        "variant_id": variant_id,
        "event_ts": decision.event_ts,
        "status": status,
        "reason": reason,
        "trigger_ts": trigger_ts,
        "entry_ts": entry_ts,
        "entry_price": entry_price,
        "payload": {
            "expected_net_bps": decision.expected_net_bps,
            "active_memories": decision.active_memories,
            "candle_range": decision.candle_range,
            "shock_z": decision.shock_z,
            "close_position": decision.close_position,
            "volatility_ratio": decision.volatility_ratio,
            "flow_imbalance": decision.flow_imbalance,
        },
    }


def _contiguous(bars: pd.DataFrame, start_index: int, end_index: int) -> bool:
    if start_index < 0 or end_index >= len(bars):
        return False
    segment = int(bars.iloc[start_index]["segment"])
    return int(bars.iloc[end_index]["segment"]) == segment


def _simulate_exit(
    bars: pd.DataFrame,
    *,
    decision: BrainDecision,
    variant_id: str,
    entry_index: int,
    entry_price: float,
) -> dict[str, Any] | None:
    end_index = entry_index + HOLDING_BARS - 1
    if not _contiguous(bars, entry_index, end_index):
        return None
    target = entry_price + decision.candle_range
    stop = entry_price - decision.candle_range
    exit_index = end_index
    exit_price = float(bars.iloc[end_index]["close"])
    reason = "time"
    for bar_index in range(entry_index, end_index + 1):
        bar = bars.iloc[bar_index]
        if float(bar["open"]) <= stop:
            exit_index, exit_price, reason = bar_index, float(bar["open"]), "stop_gap"
            break
        if float(bar["open"]) >= target:
            exit_index, exit_price, reason = bar_index, target, "target_gap"
            break
        target_hit = float(bar["high"]) >= target
        stop_hit = float(bar["low"]) <= stop
        if stop_hit:
            exit_index, exit_price = bar_index, stop
            reason = "ambiguous_stop" if target_hit else "stop"
            break
        if target_hit:
            exit_index, exit_price, reason = bar_index, target, "target"
            break
    gross_bps = float(np.log(exit_price / entry_price) * 10_000.0)
    return {
        "candidate_id": f"brain-forward:{variant_id}:{decision.event_ts.isoformat()}",
        "variant_id": variant_id,
        "event_ts": decision.event_ts,
        "entry_ts": pd.Timestamp(bars.iloc[entry_index]["ts"]),
        "exit_ts": pd.Timestamp(bars.iloc[exit_index]["ts"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "target_price": target,
        "stop_price": stop,
        "exit_reason": reason,
        "gross_bps": gross_bps,
        "net_bps": gross_bps - COST_BPS,
        "expected_net_bps": decision.expected_net_bps,
    }


def _evaluate_immediate(
    bars: pd.DataFrame,
    indexed: list[tuple[int, BrainDecision]],
    *,
    cooldown_after_stop: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variant_id = STOP_COOLDOWN_3M if cooldown_after_stop else IMMEDIATE
    events: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    last_exit = -1
    cooldown_until: pd.Timestamp | None = None
    for event_index, decision in indexed:
        entry_index = event_index + 1
        end_index = event_index + HOLDING_BARS
        if end_index >= len(bars):
            events.append(_event(variant_id, decision, "pending", reason="outcome_not_mature"))
            continue
        if not _contiguous(bars, event_index, end_index):
            events.append(_event(variant_id, decision, "suppressed", reason="missing_minute"))
            continue
        entry_ts = pd.Timestamp(bars.iloc[entry_index]["ts"])
        if entry_index <= last_exit:
            events.append(_event(variant_id, decision, "suppressed", reason="overlapping_position"))
            continue
        if cooldown_after_stop and cooldown_until is not None and entry_ts < cooldown_until:
            events.append(
                _event(
                    variant_id,
                    decision,
                    "suppressed",
                    reason="stop_cooldown",
                    trigger_ts=entry_ts,
                )
            )
            continue
        entry_price = float(bars.iloc[entry_index]["open"])
        trade = _simulate_exit(
            bars,
            decision=decision,
            variant_id=variant_id,
            entry_index=entry_index,
            entry_price=entry_price,
        )
        if trade is None:
            events.append(_event(variant_id, decision, "suppressed", reason="missing_minute"))
            continue
        trades.append(trade)
        last_exit = int(bars.index[bars["ts"].eq(trade["exit_ts"])][0])
        if cooldown_after_stop and str(trade["exit_reason"]).startswith(("stop", "ambiguous_stop")):
            cooldown_until = pd.Timestamp(trade["exit_ts"]) + pd.Timedelta(minutes=COOLDOWN_MINUTES)
        events.append(
            _event(
                variant_id,
                decision,
                "triggered",
                reason="next_minute_open",
                trigger_ts=entry_ts,
                entry_ts=entry_ts,
                entry_price=entry_price,
            )
        )
    return events, trades


def _confirmation_entry(
    bars: pd.DataFrame,
    event_index: int,
) -> tuple[int, float, str] | None:
    last_index = min(event_index + CONFIRMATION_BARS, len(bars) - 1)
    for bar_index in range(event_index + 1, last_index + 1):
        if not _contiguous(bars, event_index, bar_index):
            return None
        previous_high = float(bars.iloc[bar_index - 1]["high"])
        current = bars.iloc[bar_index]
        if float(current["open"]) > previous_high:
            return bar_index, float(current["open"]), "previous_high_gap_break"
        if float(current["high"]) > previous_high:
            return bar_index, previous_high, "previous_high_intrabar_break"
    return None


def _evaluate_confirmation(
    bars: pd.DataFrame,
    indexed: list[tuple[int, BrainDecision]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    last_exit = -1
    for event_index, decision in indexed:
        confirmation = _confirmation_entry(bars, event_index)
        confirmation_window_end = event_index + CONFIRMATION_BARS
        if confirmation is None:
            status = "pending" if confirmation_window_end >= len(bars) else "suppressed"
            reason = "confirmation_not_mature" if status == "pending" else "no_previous_high_break"
            events.append(_event(PREVIOUS_HIGH_CONFIRMATION, decision, status, reason=reason))
            continue
        entry_index, entry_price, trigger_reason = confirmation
        entry_ts = pd.Timestamp(bars.iloc[entry_index]["ts"])
        if entry_index <= last_exit:
            events.append(
                _event(
                    PREVIOUS_HIGH_CONFIRMATION,
                    decision,
                    "suppressed",
                    reason="overlapping_position",
                    trigger_ts=entry_ts,
                )
            )
            continue
        if entry_index + HOLDING_BARS - 1 >= len(bars):
            events.append(
                _event(
                    PREVIOUS_HIGH_CONFIRMATION,
                    decision,
                    "confirmed_pending",
                    reason=trigger_reason,
                    trigger_ts=entry_ts,
                    entry_ts=entry_ts,
                    entry_price=entry_price,
                )
            )
            continue
        trade = _simulate_exit(
            bars,
            decision=decision,
            variant_id=PREVIOUS_HIGH_CONFIRMATION,
            entry_index=entry_index,
            entry_price=entry_price,
        )
        if trade is None:
            events.append(_event(PREVIOUS_HIGH_CONFIRMATION, decision, "suppressed", reason="missing_minute"))
            continue
        trades.append(trade)
        last_exit = int(bars.index[bars["ts"].eq(trade["exit_ts"])][0])
        events.append(
            _event(
                PREVIOUS_HIGH_CONFIRMATION,
                decision,
                "confirmed",
                reason=trigger_reason,
                trigger_ts=entry_ts,
                entry_ts=entry_ts,
                entry_price=entry_price,
            )
        )
    return events, trades


def evaluate_paper_variants(raw: pd.DataFrame, model: FrozenUtilityMemory) -> VariantEvaluation:
    """Evaluate all entry policies against one causal signal stream."""

    features = build_feature_frame(raw)
    bars = features.loc[:, ["ts", "open", "high", "low", "close", "segment"]].copy()
    indexed: list[tuple[int, BrainDecision]] = []
    for index in range(len(features)):
        decision = model.decision_from_feature_row(features.iloc[[index]])
        if decision is not None:
            indexed.append((index, decision))
    events: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for cooldown in (False, True):
        local_events, local_trades = _evaluate_immediate(
            bars, indexed, cooldown_after_stop=cooldown
        )
        events.extend(local_events)
        trades.extend(local_trades)
    confirmation_events, confirmation_trades = _evaluate_confirmation(bars, indexed)
    events.extend(confirmation_events)
    trades.extend(confirmation_trades)
    return VariantEvaluation(
        decisions=[decision for _, decision in indexed],
        events=events,
        trades=trades,
    )
