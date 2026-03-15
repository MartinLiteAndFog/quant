from __future__ import annotations

from typing import Literal

import pandas as pd


StrategyMode = Literal["countertrend", "trendfollower"]


def strategy_for_gate(gate_on: int) -> StrategyMode:
    """
    Live mapping agreed with user:
    - Gate ON  -> countertrend (IMBA)
    - Gate OFF -> trendfollower
    """
    return "countertrend" if int(gate_on) else "trendfollower"


def trend_signals_from_imba(imba_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Build a trendfollower signal stream from raw IMBA signals.

    IMPORTANT — INTENTIONAL DESIGN (confirmed 2026-03-15):
    Both countertrend and trendfollower use the **same entry direction**
    produced by compute_imba_signals().  The strategies differ ONLY in
    their exit management (TTP/flip for countertrend, TP1/TP2 for
    trendfollower).  Signal direction is deliberately NOT inverted.
    """
    if imba_signals is None or len(imba_signals) == 0:
        return pd.DataFrame(columns=["ts", "signal", "position", "source", "sl"])

    out = imba_signals.copy()
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int)
    out = out[out["signal"] != 0].copy()
    out["position"] = out["signal"]
    out["source"] = "trend_from_imba"
    return out.reset_index(drop=True)
