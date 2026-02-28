# src/quant/predictive_coding/trade_logic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from quant.predictive_coding.config import PCConfig


@dataclass
class _Pos:
    side: int = 0               # -1 short, 0 flat, +1 long
    entry_price: float = 0.0
    entry_t: int = -1
    horizon: int = 0
    bars_in_trade: int = 0
    cooldown_left: int = 0


class TradeDecisionLayer:
    """
    v0.2 policy: Agreement gate on 5m & 15m horizons (reduces churn).

    Expects probs dict:
      probs[h] = {"mu": float, "sigma": float, "p_up": float, "price_level": float, ...}

    Returns:
      signal: -1/0/+1
      events: list[dict] with keys:
        event: "entry" | "sl_exit" | "tp_exit" | "timeout_exit" | "flip_exit"
        side: int
        price: float
        pnl_pct: float (raw pnl, no costs)
        seq: int
        horizon: int
        edge: float (decision edge in return space)
    """

    def __init__(self, cfg: PCConfig) -> None:
        self.cfg = cfg
        self.pos = _Pos()
        self.seq = 0

    # ---------- backward-compatible state aliases ----------

    @property
    def position(self) -> int:
        return int(self.pos.side)

    @position.setter
    def position(self, v: int) -> None:
        self.pos.side = int(v)

    @property
    def entry_price(self) -> float:
        return float(self.pos.entry_price)

    @entry_price.setter
    def entry_price(self, v: float) -> None:
        self.pos.entry_price = float(v)

    @property
    def entry_bar(self) -> int:
        return int(self.pos.entry_t)

    @entry_bar.setter
    def entry_bar(self, v: int) -> None:
        self.pos.entry_t = int(v)

    @property
    def chosen_horizon(self) -> int:
        return int(self.pos.horizon)

    @chosen_horizon.setter
    def chosen_horizon(self, v: int) -> None:
        self.pos.horizon = int(v)

    @property
    def cooldown_remaining(self) -> int:
        return int(self.pos.cooldown_left)

    @cooldown_remaining.setter
    def cooldown_remaining(self, v: int) -> None:
        self.pos.cooldown_left = int(v)

    # ---------- helpers ----------

    def _cost(self) -> float:
        # roundtrip cost in return units
        return (self.cfg.fee_bps + self.cfg.slippage_bps) / 10_000.0

    def _min_edge(self) -> float:
        return self.cfg.min_edge_bps / 10_000.0

    def _pnl_pct(self, side: int, entry: float, px: float) -> float:
        if side == 0 or entry <= 0:
            return 0.0
        r = (px / entry) - 1.0
        return r if side > 0 else -r

    def _emit(self, event: str, side: int, price: float, pnl_pct: float, horizon: int, edge: float = 0.0) -> Dict[str, Any]:
        self.seq += 1
        return {
            "event": event,
            "side": int(side),
            "price": float(price),
            "pnl_pct": float(pnl_pct),
            "seq": int(self.seq),
            "horizon": int(horizon),
            "edge": float(edge),
        }

    # ---------- decision logic ----------

    def _agreement_signal(self, probs: Dict[int, Dict[str, float]]) -> Tuple[int, int, float]:
        """
        Returns (signal_side, chosen_h, edge).
        Agreement: both 5m and 15m must exceed thresholds in same direction.
        Chooses horizon=15 by default (more stable), unless 5 is MUCH stronger (not implemented here).
        """
        cfg = self.cfg
        cost = self._cost()
        min_edge = self._min_edge()

        # required horizons
        if 5 not in probs or 15 not in probs:
            return 0, 0, 0.0

        p5 = probs[5]["p_up"]
        p15 = probs[15]["p_up"]
        mu15 = probs[15]["mu"]
        sig15 = probs[15]["sigma"]
        z15 = (mu15 / sig15) if sig15 > 0 else 0.0

        # optional veto: if 60m screams opposite, skip (helps avoid trading into larger drift)
        veto = False
        if 60 in probs:
            p60 = probs[60]["p_up"]
            # if long setup but 60m is strongly short, veto; vice versa
            # use flip_margin as "strong" threshold
            if (p5 > 0.5 + cfg.margin and p15 > 0.5 + cfg.margin) and (p60 < 0.5 - cfg.flip_margin):
                veto = True
            if (p5 < 0.5 - cfg.margin and p15 < 0.5 - cfg.margin) and (p60 > 0.5 + cfg.flip_margin):
                veto = True
        if veto:
            return 0, 0, 0.0

        # LONG agreement
        if (p5 > 0.5 + cfg.margin) and (p15 > 0.5 + cfg.margin) and (z15 > cfg.z_min):
            edge = (mu15 - cost - min_edge)
            if edge > 0:
                return +1, 15, float(edge)

        # SHORT agreement
        if (p5 < 0.5 - cfg.margin) and (p15 < 0.5 - cfg.margin) and (z15 < -cfg.z_min):
            edge = (-mu15 - cost - min_edge)
            if edge > 0:
                return -1, 15, float(edge)

        return 0, 0, 0.0

    # ---------- state machine ----------

    def update(
        self,
        probs: Dict[int, Dict[str, float]],
        close: float,
        t: Optional[int] = None,
        *,
        bar_idx: Optional[int] = None,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        cfg = self.cfg
        close = float(close)
        if t is None:
            if bar_idx is None:
                raise TypeError("update() missing time index argument (t or bar_idx)")
            t = int(bar_idx)
        else:
            t = int(t)
        events: List[Dict[str, Any]] = []

        # cooldown tick
        if self.pos.cooldown_left > 0:
            self.pos.cooldown_left -= 1

        # Backward-compatible behavior: if flat and in cooldown, emit no signal.
        if self.pos.side == 0 and self.pos.cooldown_left > 0:
            return 0, events

        # compute desired signal
        sig, best_h, edge = self._agreement_signal(probs)

        # bars in trade: prefer t-entry_t semantics (compat with older runner/tests)
        if self.pos.side != 0:
            if self.pos.entry_t >= 0:
                self.pos.bars_in_trade = max(self.pos.bars_in_trade, t - self.pos.entry_t)
            else:
                self.pos.bars_in_trade += 1

        # ----- exits first -----
        if self.pos.side != 0:
            pnl = self._pnl_pct(self.pos.side, self.pos.entry_price, close)

            # 1) stop-loss
            if pnl <= -cfg.sl_pct:
                events.append(self._emit("sl_exit", self.pos.side, close, pnl, self.pos.horizon))
                self.pos = _Pos(cooldown_left=cfg.cooldown_bars)
                return sig, events

            # 2) take-profit
            if pnl >= cfg.tp_pct:
                events.append(self._emit("tp_exit", self.pos.side, close, pnl, self.pos.horizon))
                self.pos = _Pos(cooldown_left=cfg.cooldown_bars)
                return sig, events

            # 3) timeout
            timeout_bars = cfg.timeout_bars if getattr(cfg, "timeout_bars", None) is not None else self.pos.horizon
            if timeout_bars and self.pos.bars_in_trade >= int(timeout_bars):
                events.append(self._emit("timeout_exit", self.pos.side, close, pnl, self.pos.horizon))
                self.pos = _Pos(cooldown_left=cfg.cooldown_bars)
                return sig, events

            # 4) flip (only if strong opposite AND not in cooldown AND held at least a bit)
            if self.pos.cooldown_left == 0 and sig != 0 and sig != self.pos.side:
                # strong flip conditions: use flip_margin + stronger z
                if best_h in probs:
                    p = probs[best_h]["p_up"]
                    mu = probs[best_h]["mu"]
                    s = probs[best_h]["sigma"]
                    z = (mu / s) if s > 0 else 0.0

                    strong = False
                    if sig > 0:
                        strong = (p > 0.5 + cfg.flip_margin) and (z > max(cfg.z_min, 0.50))
                    else:
                        strong = (p < 0.5 - cfg.flip_margin) and (z < -max(cfg.z_min, 0.50))

                    # require at least 2 bars held to avoid instant ping-pong
                    if strong and self.pos.bars_in_trade >= 2:
                        # exit old
                        events.append(self._emit("flip_exit", self.pos.side, close, pnl, self.pos.horizon))
                        # enter new
                        self.pos = _Pos(side=sig, entry_price=close, entry_t=t, horizon=best_h, bars_in_trade=0, cooldown_left=cfg.cooldown_bars)
                        events.append(self._emit("entry", self.pos.side, close, 0.0, self.pos.horizon, edge=edge))
                        return sig, events

        # ----- entries -----
        if self.pos.side == 0 and self.pos.cooldown_left == 0:
            if sig != 0:
                self.pos = _Pos(side=sig, entry_price=close, entry_t=t, horizon=best_h, bars_in_trade=0, cooldown_left=0)
                events.append(self._emit("entry", self.pos.side, close, 0.0, self.pos.horizon, edge=edge))

        return sig, events