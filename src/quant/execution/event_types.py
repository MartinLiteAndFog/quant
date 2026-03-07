from __future__ import annotations

from typing import Any, Dict, Literal, Optional, TypedDict


EventFamily = Literal["signal_events", "action_events", "execution_events"]


class BaseEvent(TypedDict, total=False):
    event_id: str
    event_family: EventFamily
    strategy: str
    symbol: str
    venue: str
    ts: str
    seq: int
    position_before: int
    position_after: int
    engine_mode_before: str
    engine_mode_after: str
    reason_code: str
    source_event_id: Optional[str]
    source_signal_event_id: Optional[str]
    blocked: bool
    block_reason: Optional[str]


class SignalEvent(BaseEvent, total=False):
    signal: int
    signal_side: str
    signal_family: str
    signal_kind: str


class ActionEvent(BaseEvent, total=False):
    engine_action: str
    action_side: str


class ExecutionEvent(BaseEvent, total=False):
    execution_kind: str
    order_action: str


EventDict = Dict[str, Any]
