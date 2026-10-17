"""Experimental, isolated Tradingade-style entry filters and signal-only exits.

Not a verified replica of Tradingade. Signals are evaluated ONLY after confirmed
bars; a live/paper execution layer must execute at the next real-market price.
No intrabar broker stop is created or simulated by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EntryFilter:
    rsi_period: int = 14
    long_above: float = 55.0
    short_below: float = 75.0


def wilder_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder/TradingView-like RSI for *completed* close samples (RMA)."""
    if period < 1:
        raise ValueError("period must be positive")
    s = pd.to_numeric(closes, errors="coerce").astype(float)
    delta = s.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    def rma(x: pd.Series) -> pd.Series:
        values = x.to_numpy(dtype=float)
        out = np.full(len(values), np.nan)
        if len(values) <= period:
            return pd.Series(out, index=x.index)
        seed = np.nanmean(values[1:period + 1])
        out[period] = seed
        for j in range(period + 1, len(values)):
            out[j] = (out[j-1] * (period-1) + values[j]) / period
        return pd.Series(out, index=x.index)

    up, down = rma(gains), rma(losses)
    rs = up / down.replace(0, np.nan)
    output = 100 - 100 / (1 + rs)
    output = output.mask((down == 0) & (up > 0), 100.0)
    output = output.mask((down == 0) & (up == 0), 50.0)
    return output


def confirmed_30m_rsi(minute_ohlc: pd.DataFrame, *, period: int = 14) -> pd.DataFrame:
    """Return [available_at, rsi30] from completed thirty-minute close candles.

    Input timestamps are UTC one-minute BAR OPENS. Incomplete buckets yield NaN.
    The RSI is first available at the end, not at the start, of its 30m candle.
    """
    df = minute_ohlc[["ts", "close"]].copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    s = df.set_index("ts")["close"]
    grouped = s.resample("30min", label="left", closed="left")
    closes = grouped.last()
    counts = grouped.count()
    closes = closes.where(counts == 30)
    rsi = wilder_rsi(closes, period)
    return pd.DataFrame({"available_at": rsi.index + pd.Timedelta(minutes=30),
                         "rsi30": rsi.to_numpy()})


def attach_last_confirmed_rsi(decision_events: pd.DataFrame,
                              confirmed_rsi: pd.DataFrame) -> pd.DataFrame:
    """Causal as-of join: decision_ts is signal confirmation, not pivot timestamp."""
    events = decision_events.copy()
    events["decision_ts"] = pd.to_datetime(events["decision_ts"], utc=True)
    avail = confirmed_rsi.copy()
    avail["available_at"] = pd.to_datetime(avail["available_at"], utc=True)
    events["_order"] = np.arange(len(events))
    result = pd.merge_asof(events.sort_values("decision_ts"),
                           avail.sort_values("available_at"),
                           left_on="decision_ts", right_on="available_at",
                           direction="backward", allow_exact_matches=True)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def is_entry_allowed(side: int, rsi: float, filt: EntryFilter = EntryFilter()) -> bool:
    """RSI is checked on ENTRY only; it never forces an exit."""
    return bool(np.isfinite(rsi) and ((side == 1 and rsi > filt.long_above)
                                     or (side == -1 and rsi < filt.short_below)))


@dataclass(frozen=True)
class SignalExitParams:
    stop_loss_pct: float = 0.011
    trail_activation_pct: float = 0.0
    trail_deviation_pct: float = 0.008

    def __post_init__(self) -> None:
        if not 0 < self.stop_loss_pct < 1:
            raise ValueError("stop_loss_pct must be between zero and one")
        if not 0 < self.trail_deviation_pct < 1:
            raise ValueError("trail_deviation_pct must be between zero and one")
        if self.trail_activation_pct < 0:
            raise ValueError("trail_activation_pct must not be negative")


ExitReason = Literal["SL_SIGNAL", "TRAILING_TP_SIGNAL"]


@dataclass
class SignalOnlyTrailing:
    """One open position; only confirmed synthetic chart closes form exit signals.

    Returned exit is a STRATEGY DECISION, not a venue order/fill. The caller
    prices it at the next eligible REAL 1m open after the 5m source bar closes.

    Tradingade's exact TTP-zero activation and intrabar semantics still require
    calibration against user-supplied trade exports before claiming parity.
    """
    side: int
    chart_entry_price: float
    params: SignalExitParams = SignalExitParams()
    best_chart_price: Optional[float] = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.side not in (-1, 1):
            raise ValueError("side must be +1 (LONG) or -1 (SHORT)")
        if self.chart_entry_price <= 0:
            raise ValueError("chart_entry_price must be positive")
        self.best_chart_price = float(self.chart_entry_price)

    def mark(self, confirmed_chart_close: float) -> Optional[ExitReason]:
        if self.closed:
            raise RuntimeError("Position already closed")
        price = float(confirmed_chart_close)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("positive finite chart close required")
        if self.side == 1:
            self.best_chart_price = max(self.best_chart_price, price)
            stop_hit = price <= self.chart_entry_price * (1 - self.params.stop_loss_pct)
            activated = self.best_chart_price >= self.chart_entry_price * (1 + self.params.trail_activation_pct)
            trail_hit = activated and price <= self.best_chart_price * (1 - self.params.trail_deviation_pct)
        else:
            self.best_chart_price = min(self.best_chart_price, price)
            stop_hit = price >= self.chart_entry_price * (1 + self.params.stop_loss_pct)
            activated = self.best_chart_price <= self.chart_entry_price * (1 - self.params.trail_activation_pct)
            trail_hit = activated and price >= self.best_chart_price * (1 + self.params.trail_deviation_pct)
        # If both levels were crossed between confirmed bars, chronology is
        # unknowable from chart closes alone: conservatively categorize as SL.
        result: Optional[ExitReason] = "SL_SIGNAL" if stop_hit else (
            "TRAILING_TP_SIGNAL" if trail_hit else None)
        if result is not None:
            self.closed = True
        return result
