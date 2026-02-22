# src/quant/execution/oms.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal


Side = Literal["buy", "sell"]
Reason = Literal["entry", "tp_flip", "sl_exit", "flatten"]


@dataclass(frozen=True)
class OmsDefaults:
    """
    Maker-first + entry-ladder defaults (your agreed parameters).

    All pct values are decimals:
      0.0002 = 0.02% = 2 bps
    """
    # Entry ladder (rebound entry)
    l1_pct: float = 0.0002       # 0.02%
    l2_pct: float = 0.0005       # 0.05%
    entry_timeout_s: int = 60
    entry_fallback_off_pct: float = 0.0003  # 0.03%

    # TP/Flip exit (maker-first only if "calm")
    tp_timeout_s: int = 30
    tp_fallback_off_pct: float = 0.0003     # 0.03%

    # SL exit (always fast)
    sl_marketable_off_pct: float = 0.0006   # 0.06%

    # Calm gate (based on 1m range pct proxy)
    # maker_ok if range_1m_pct <= vol_gate_thr
    vol_gate_thr: float = 0.0014            # ~p50 from your stats
    vol_gate_hard_off: float = 0.0038       # ~p90; above this never maker

    # Requote behavior (cancel/replace)
    reprice_every_s_entry: int = 15
    max_requotes_entry: int = 4
    reprice_every_s_tp: int = 10
    max_requotes_tp: int = 3


class BrokerAPI:
    """
    Minimal interface your KuCoin client/executor should implement.
    Plug your real exchange adapter here.

    IMPORTANT: This file does not talk to any exchange by itself.
    """
    def get_best_bid_ask(self, symbol: str) -> tuple[float, float]:
        raise NotImplementedError

    def get_1m_range_pct_proxy(self, symbol: str) -> Optional[float]:
        """
        Return current 1m range proxy: (high-low)/close, or None if unknown.
        Your live implementation can compute it from latest 1m candle.
        """
        raise NotImplementedError

    def get_position(self, symbol: str) -> float:
        """Signed position size: >0 long, <0 short, 0 flat."""
        raise NotImplementedError

    def cancel_all(self, symbol: str) -> None:
        raise NotImplementedError

    def place_limit(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        post_only: bool,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        """Return order_id."""
        raise NotImplementedError

    def place_marketable_limit(
        self,
        symbol: str,
        side: Side,
        qty: float,
        limit_price: float,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        """
        A limit that is intentionally marketable (crosses spread),
        so it fills like taker but has a price cap.
        """
        raise NotImplementedError

    def wait_filled(self, symbol: str, order_id: str, timeout_s: int) -> bool:
        """Return True if filled within timeout."""
        raise NotImplementedError


@dataclass
class OmsResult:
    ok: bool
    mode: str  # e.g. "L2", "L1", "FB", "PO", "MK"
    details: Dict[str, Any]


class MakerFirstOMS:
    """
    Maker-first OMS with:
      - entry ladder (2 passive limits) + fallback
      - tp/flip maker-first in calm regime + fallback
      - sl always fast (marketable)
      - flatten-first flip workflow
    """
    def __init__(self, broker: BrokerAPI, cfg: Optional[OmsDefaults] = None):
        self.broker = broker
        self.cfg = cfg or OmsDefaults()

    def enter(self, symbol: str, side: Literal["long", "short"], qty: float) -> OmsResult:
        want_side: Side = "buy" if side == "long" else "sell"
        return self._entry_ladder(symbol=symbol, side=want_side, qty=qty, reason="entry")

    def exit_tp_or_flip(self, symbol: str, side: Literal["long", "short"], qty: float, flip_to: Optional[Literal["long", "short"]] = None) -> OmsResult:
        flatten_side: Side = "sell" if side == "long" else "buy"
        flat = self._tp_maker_first_or_fallback(symbol=symbol, side=flatten_side, qty=qty, reduce_only=True, reason="tp_flip")
        if not flat.ok:
            return flat

        if flip_to is None:
            return flat

        if not self._wait_flat(symbol, timeout_s=30):
            return OmsResult(False, "FLAT_TIMEOUT", {"symbol": symbol})

        return self.enter(symbol, flip_to, qty)

    def exit_sl(self, symbol: str, side: Literal["long", "short"], qty: float) -> OmsResult:
        exit_side: Side = "sell" if side == "long" else "buy"
        return self._sl_fast(symbol=symbol, side=exit_side, qty=qty)

    def _maker_ok(self, symbol: str) -> bool:
        r = self.broker.get_1m_range_pct_proxy(symbol)
        if r is None:
            return False
        if r > self.cfg.vol_gate_hard_off:
            return False
        return r <= self.cfg.vol_gate_thr

    def _entry_ladder(self, symbol: str, side: Side, qty: float, reason: Reason) -> OmsResult:
        bid, ask = self.broker.get_best_bid_ask(symbol)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (ask or bid)

        if not mid or mid <= 0:
            return self._fallback_marketable(symbol, side, qty, ref_price=1.0, off_pct=self.cfg.entry_fallback_off_pct, reduce_only=False, mode="FB_NOQUOTE")

        ref = ask if side == "sell" else bid
        if not ref or ref <= 0:
            ref = mid

        if side == "buy":
            p1 = ref * (1.0 - self.cfg.l1_pct)
            p2 = ref * (1.0 - self.cfg.l2_pct)
        else:
            p1 = ref * (1.0 + self.cfg.l1_pct)
            p2 = ref * (1.0 + self.cfg.l2_pct)

        targets = [("L2", p2), ("L1", p1)]

        t0 = time.time()
        requotes = 0

        while True:
            if not self._maker_ok(symbol):
                return self._fallback_marketable(symbol, side, qty, ref_price=ref, off_pct=self.cfg.entry_fallback_off_pct, reduce_only=False, mode="FB_VOL")

            for tag, price in targets:
                cid = f"{reason}:{tag}:{int(time.time()*1000)}"
                oid = self.broker.place_limit(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    price=float(price),
                    post_only=True,
                    reduce_only=False,
                    client_id=cid,
                )
                filled = self.broker.wait_filled(symbol, oid, timeout_s=self.cfg.reprice_every_s_entry)
                if filled:
                    return OmsResult(True, tag, {"symbol": symbol, "side": side, "qty": qty, "price": float(price), "client_id": cid, "order_id": oid})

                self.broker.cancel_all(symbol)

            if (time.time() - t0) >= self.cfg.entry_timeout_s:
                return self._fallback_marketable(symbol, side, qty, ref_price=ref, off_pct=self.cfg.entry_fallback_off_pct, reduce_only=False, mode="FB_TIMEOUT")

            requotes += 1
            if requotes >= self.cfg.max_requotes_entry:
                return self._fallback_marketable(symbol, side, qty, ref_price=ref, off_pct=self.cfg.entry_fallback_off_pct, reduce_only=False, mode="FB_REQUOTE_MAX")

    def _tp_maker_first_or_fallback(self, symbol: str, side: Side, qty: float, reduce_only: bool, reason: Reason) -> OmsResult:
        bid, ask = self.broker.get_best_bid_ask(symbol)
        ref = (ask if side == "sell" else bid) or (bid if side == "sell" else ask)

        if not ref or ref <= 0:
            return self._fallback_marketable(symbol, side, qty, ref_price=1.0, off_pct=self.cfg.tp_fallback_off_pct, reduce_only=reduce_only, mode="TP_FB_NOQUOTE")

        t0 = time.time()
        requotes = 0

        while True:
            if self._maker_ok(symbol):
                bid, ask = self.broker.get_best_bid_ask(symbol)
                px = (ask if side == "sell" else bid) or ref

                cid = f"{reason}:PO:{int(time.time()*1000)}"
                oid = self.broker.place_limit(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    price=float(px),
                    post_only=True,
                    reduce_only=reduce_only,
                    client_id=cid,
                )
                filled = self.broker.wait_filled(symbol, oid, timeout_s=self.cfg.reprice_every_s_tp)
                if filled:
                    return OmsResult(True, "PO", {"symbol": symbol, "side": side, "qty": qty, "price": float(px), "client_id": cid, "order_id": oid})

                self.broker.cancel_all(symbol)

                if (time.time() - t0) >= self.cfg.tp_timeout_s or requotes >= self.cfg.max_requotes_tp:
                    return self._fallback_marketable(symbol, side, qty, ref_price=px, off_pct=self.cfg.tp_fallback_off_pct, reduce_only=reduce_only, mode="TP_FB_TIMEOUT")

                requotes += 1
                continue

            return self._fallback_marketable(symbol, side, qty, ref_price=ref, off_pct=self.cfg.tp_fallback_off_pct, reduce_only=reduce_only, mode="TP_FB_VOL")

    def _sl_fast(self, symbol: str, side: Side, qty: float) -> OmsResult:
        bid, ask = self.broker.get_best_bid_ask(symbol)
        ref = bid if side == "sell" else ask
        if not ref or ref <= 0:
            ref = (bid + ask) / 2.0 if (bid and ask) else (ask or bid or 1.0)
        return self._fallback_marketable(symbol, side, qty, ref_price=float(ref), off_pct=self.cfg.sl_marketable_off_pct, reduce_only=True, mode="SL_MK")

    def _fallback_marketable(self, symbol: str, side: Side, qty: float, ref_price: float, off_pct: float, reduce_only: bool, mode: str) -> OmsResult:
        ref_price = float(ref_price) if ref_price and ref_price > 0 else 1.0
        limit_px = ref_price * (1.0 + off_pct) if side == "buy" else ref_price * (1.0 - off_pct)

        cid = f"{mode}:{int(time.time()*1000)}"
        oid = self.broker.place_marketable_limit(
            symbol=symbol,
            side=side,
            qty=qty,
            limit_price=float(limit_px),
            reduce_only=reduce_only,
            client_id=cid,
        )
        filled = self.broker.wait_filled(symbol, oid, timeout_s=30)
        return OmsResult(bool(filled), mode, {"symbol": symbol, "side": side, "qty": qty, "limit_px": float(limit_px), "client_id": cid, "order_id": oid, "reduce_only": reduce_only})

    def _wait_flat(self, symbol: str, timeout_s: int = 30) -> bool:
        t0 = time.time()
        while (time.time() - t0) < timeout_s:
            pos = float(self.broker.get_position(symbol))
            if abs(pos) < 1e-12:
                return True
            time.sleep(0.25)
        return False
