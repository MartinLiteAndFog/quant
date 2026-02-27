from __future__ import annotations
from quant.predictive_coding.config import PCConfig


class TradeDecisionLayer:
    def __init__(self, cfg: PCConfig) -> None:
        self.cfg = cfg
        self.position: int = 0       # -1, 0, 1
        self.entry_price: float = 0.0
        self.entry_bar: int = 0
        self.chosen_horizon: int = 0
        self.cooldown_remaining: int = 0
        self.trade_seq: int = 0

    def update(
        self,
        probs: dict[int, dict[str, float]],
        price: float,
        bar_idx: int,
    ) -> tuple[int, list[dict]]:
        cfg = self.cfg
        events: list[dict] = []
        cost = cfg.total_cost
        min_edge = cfg.min_edge_bps / 10_000

        # --- Cooldown ---
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.position != 0:
                events.extend(self._check_exits(price, bar_idx, probs))
            return self.position, events

        # --- If in position: check exits first ---
        if self.position != 0:
            exit_events = self._check_exits(price, bar_idx, probs)
            events.extend(exit_events)
            if self.position == 0:
                return 0, events

            flip_events = self._check_flip(price, bar_idx, probs)
            events.extend(flip_events)
            return self.position, events

        # --- If flat: check entry ---
        best_score = -1.0
        best_dir = 0
        best_h = 0

        for h, p in probs.items():
            mu = p["mu"]
            z = p["z"]
            p_up = p["p_up"]

            if mu > cost + min_edge and p_up > 0.5 + cfg.margin and z > cfg.z_min:
                score = (mu - cost) * (2 * p_up - 1)
                if score > best_score:
                    best_score = score
                    best_dir = 1
                    best_h = h

            if -mu > cost + min_edge and p_up < 0.5 - cfg.margin and z < -cfg.z_min:
                score = (-mu - cost) * (1 - 2 * p_up)
                if score > best_score:
                    best_score = score
                    best_dir = -1
                    best_h = h

        if best_dir != 0:
            self.position = best_dir
            self.entry_price = price
            self.entry_bar = bar_idx
            self.chosen_horizon = best_h
            self.trade_seq += 1
            events.append({
                "event": "entry",
                "side": best_dir,
                "price": price,
                "bar_idx": bar_idx,
                "horizon": best_h,
                "edge": best_score,
                "seq": self.trade_seq,
                "pnl_pct": 0.0,
            })

        return self.position, events

    def _check_exits(
        self, price: float, bar_idx: int, probs: dict[int, dict[str, float]]
    ) -> list[dict]:
        cfg = self.cfg
        events: list[dict] = []
        if self.position == 0:
            return events

        pnl_pct = self.position * (price - self.entry_price) / self.entry_price
        bars_held = bar_idx - self.entry_bar
        timeout = self.chosen_horizon if self.chosen_horizon > 0 else 60

        exit_event = None
        if pnl_pct < -cfg.sl_pct:
            exit_event = "sl_exit"
        elif pnl_pct > cfg.tp_pct:
            exit_event = "tp_exit"
        elif bars_held >= timeout:
            exit_event = "timeout_exit"

        if exit_event is not None:
            events.append({
                "event": exit_event,
                "side": self.position,
                "price": price,
                "bar_idx": bar_idx,
                "pnl_pct": pnl_pct,
                "seq": self.trade_seq,
                "horizon": self.chosen_horizon,
                "edge": 0.0,
            })
            self.position = 0
            self.entry_price = 0.0
            self.cooldown_remaining = cfg.cooldown_bars

        return events

    def _check_flip(
        self, price: float, bar_idx: int, probs: dict[int, dict[str, float]]
    ) -> list[dict]:
        cfg = self.cfg
        events: list[dict] = []
        cost = cfg.total_cost
        min_edge = cfg.min_edge_bps / 10_000

        for h, p in probs.items():
            mu = p["mu"]
            z = p["z"]
            p_up = p["p_up"]

            should_flip = False
            new_dir = 0

            if self.position == 1:
                if (
                    -mu > cost + min_edge
                    and p_up < 0.5 - cfg.flip_margin
                    and z < -cfg.z_flip_min
                ):
                    should_flip = True
                    new_dir = -1

            elif self.position == -1:
                if (
                    mu > cost + min_edge
                    and p_up > 0.5 + cfg.flip_margin
                    and z > cfg.z_flip_min
                ):
                    should_flip = True
                    new_dir = 1

            if should_flip:
                pnl_pct = self.position * (price - self.entry_price) / self.entry_price
                events.append({
                    "event": "flip_exit",
                    "side": self.position,
                    "price": price,
                    "bar_idx": bar_idx,
                    "pnl_pct": pnl_pct,
                    "seq": self.trade_seq,
                    "horizon": self.chosen_horizon,
                    "edge": 0.0,
                })
                self.position = new_dir
                self.entry_price = price
                self.entry_bar = bar_idx
                self.chosen_horizon = h
                self.trade_seq += 1
                events.append({
                    "event": "entry",
                    "side": new_dir,
                    "price": price,
                    "bar_idx": bar_idx,
                    "horizon": h,
                    "edge": 0.0,
                    "seq": self.trade_seq,
                    "pnl_pct": 0.0,
                })
                self.cooldown_remaining = cfg.cooldown_bars
                break

        return events
