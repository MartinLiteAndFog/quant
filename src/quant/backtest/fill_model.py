# src/quant/backtest/fill_model.py
"""
Maker/taker fill model for backtest trades.

Where do bps come from?
- Fees (e.g. KuCoin): maker 4 bps roundtrip, taker 12 bps roundtrip. Set fee_maker_bps_roundtrip and
  fee_taker_bps_roundtrip to apply per-trade fees from fill_mode_entry/fill_mode_exit (L1/L2/PO=maker, FB/SL_MK=taker).
  Otherwise a single fee_bps_roundtrip is used for all trades.
- Fill quality (L1/L2/FB, tp_maker, sl_taker): offset vs reference price (limit better, fallback worse).
  This is NOT slippage; it's "did we get filled as maker or taker?".
- Slippage (market impact): not in here by default. Use slippage_bps_roundtrip to add extra cost per trade.

Current model: L1/L2 = price improvement (maker), FB = worse (taker); SL = taker. Fee subtracted once per trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FillModelParams:
    """Aligned with OmsDefaults; bps as decimals."""
    # Entry ladder (improvement when maker fills)
    l1_bps: float = 0.0002   # 2 bps better
    l2_bps: float = 0.0005   # 5 bps better
    entry_fallback_bps: float = 0.0003   # 3 bps worse when fallback
    # Probabilities for entry (if None, use equal or calibrate from live stats later)
    p_entry_l1: float = 0.35
    p_entry_l2: float = 0.35
    p_entry_fb: float = 0.30
    # TP/Flip exit
    tp_maker_bps: float = 0.0002
    tp_fallback_bps: float = 0.0003
    p_tp_maker: float = 0.50
    p_tp_fb: float = 0.50
    # SL always taker
    sl_taker_bps: float = 0.0006   # 6 bps worse
    # Fee: single value for all trades, or per fill-mode (KuCoin-style)
    fee_bps_roundtrip: float = 0.0015   # 15 bps if not using maker/taker split
    fee_maker_bps_roundtrip: float = 0.0   # e.g. 0.0004 (4 bps); if > 0 with fee_taker use per-trade fee
    fee_taker_bps_roundtrip: float = 0.0   # e.g. 0.0012 (12 bps)
    slippage_bps_roundtrip: float = 0.0  # optional: extra cost per trade (market impact)


def _is_sl_or_regime(exit_event: str) -> bool:
    e = (exit_event or "").strip().lower()
    return e in ("sl_exit", "regime_exit", "be_exit")


def _is_tp_or_flip(exit_event: str) -> bool:
    e = (exit_event or "").strip().lower()
    return e in ("tp_exit", "signal_flip_exit", "tp1_exit", "tp2_exit", "signal_exit")


def apply_fill_model(
    trades: pd.DataFrame,
    params: Optional[FillModelParams] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Adjust entry/exit prices by maker/taker model; add entry_px_adj, exit_px_adj, pnl_pct_adj.

    Expects columns: entry_px, exit_px, side (1 long, -1 short), and exit_event.
    Optional: strategy ('on'/'off') to treat TP2 exits consistently.
    """
    p = params or FillModelParams()
    t = trades.copy()
    if "entry_px" not in t.columns or "exit_px" not in t.columns or "side" not in t.columns:
        raise ValueError("trades need columns entry_px, exit_px, side")
    exit_event = t.get("exit_event", pd.Series(["tp_exit"] * len(t)))
    n = len(t)
    rng = np.random.default_rng(seed)

    # Entry fill mode: L1 / L2 / FB
    u = rng.random(n)
    entry_mode = np.where(u < p.p_entry_l1, "L1", np.where(u < p.p_entry_l1 + p.p_entry_l2, "L2", "FB"))
    # Entry price adjustment: long pays less with maker (subtract bps), short pays less (add bps to ref = lower pay)
    entry_bps = np.where(entry_mode == "L1", -p.l1_bps, np.where(entry_mode == "L2", -p.l2_bps, p.entry_fallback_bps))
    side = np.asarray(t["side"].astype(float).values)
    entry_adj = 1.0 + entry_bps * np.sign(side)  # long +1: improvement = negative bps -> factor < 1
    t["entry_px_adj"] = t["entry_px"].astype(float).values * entry_adj
    t["fill_mode_entry"] = list(entry_mode)

    # Exit: SL/regime/BE -> taker (worse); TP/flip -> maker (better) or FB (worse)
    sl_mask = exit_event.astype(str).str.strip().str.lower().apply(_is_sl_or_regime).values
    tp_mask = exit_event.astype(str).str.strip().str.lower().apply(_is_tp_or_flip).values
    u_ex = rng.random(n)
    exit_maker = tp_mask & (u_ex < p.p_tp_maker)
    exit_bps_worse = np.where(sl_mask, p.sl_taker_bps, np.where(exit_maker, 0.0, p.tp_fallback_bps))
    exit_bps_better = np.where(exit_maker, p.tp_maker_bps, 0.0)
    # Long exit (sell): better = higher px -> 1+bps; worse = lower -> 1-bps. Short: better = lower buy -> 1-bps; worse = higher -> 1+bps
    exit_adj = 1.0 + np.sign(side) * (exit_bps_better - exit_bps_worse)
    t["exit_px_adj"] = t["exit_px"].astype(float).values * exit_adj
    t["fill_mode_exit"] = np.where(sl_mask, "SL_MK", np.where(exit_maker, "PO", "FB"))

    entry_px_a = t["entry_px_adj"].astype(float).values
    exit_px_a = t["exit_px_adj"].astype(float).values
    gross = side * (exit_px_a - entry_px_a) / np.maximum(np.abs(entry_px_a), 1e-12)

    # Fee: per-trade from fill mode (maker 4 bps / taker 12 bps rt) or single fee_bps_roundtrip
    if p.fee_maker_bps_roundtrip > 0 and p.fee_taker_bps_roundtrip > 0:
        entry_maker = (np.asarray(t["fill_mode_entry"]) == "L1") | (np.asarray(t["fill_mode_entry"]) == "L2")
        exit_maker = np.asarray(t["fill_mode_exit"]) == "PO"
        fee_rt = np.where(
            entry_maker & exit_maker,
            p.fee_maker_bps_roundtrip,
            np.where(
                (~entry_maker) & (~exit_maker),
                p.fee_taker_bps_roundtrip,
                (p.fee_maker_bps_roundtrip + p.fee_taker_bps_roundtrip) * 0.5,
            ),
        )
    else:
        fee_rt = np.full(n, p.fee_bps_roundtrip)
    cost = fee_rt + p.slippage_bps_roundtrip
    t["pnl_pct_adj"] = gross - cost
    return t


def apply_fill_model_from_oms_defaults(
    trades: pd.DataFrame,
    fee_bps_roundtrip: float = 15.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Use OmsDefaults-style bps; probabilities are placeholders until you calibrate from live."""
    from quant.execution.oms import OmsDefaults
    cfg = OmsDefaults()
    params = FillModelParams(
        l1_bps=cfg.l1_pct,
        l2_bps=cfg.l2_pct,
        entry_fallback_bps=cfg.entry_fallback_off_pct,
        tp_maker_bps=cfg.tp_fallback_off_pct,
        tp_fallback_bps=cfg.tp_fallback_off_pct,
        sl_taker_bps=cfg.sl_marketable_off_pct,
        fee_bps_roundtrip=fee_bps_roundtrip / 10_000.0,
    )
    return apply_fill_model(trades, params=params, seed=seed)
