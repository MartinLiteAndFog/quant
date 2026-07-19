from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from quant.execution.event_store import load_live_renko_bricks_from_postgres
from quant.state_space.config import StateSpaceConfig
from quant.state_space.pipeline import compute_state_space


PROFILE_COUNTERTREND = "countertrend"
PROFILE_COUNTERTREND_SL_REVERSE = "countertrend_sl_reverse"
PROFILE_PC3AXIS = "pc3axis"
PROFILE_CANONICAL = "canonical"
SUPPORTED_PROFILES = {
    PROFILE_CANONICAL,
    PROFILE_COUNTERTREND,
    PROFILE_COUNTERTREND_SL_REVERSE,
    PROFILE_PC3AXIS,
}


def active_profile() -> str:
    profile = str(os.getenv("BOT_PROFILE", PROFILE_CANONICAL)).strip().lower()
    if profile not in SUPPORTED_PROFILES | {PROFILE_CANONICAL}:
        raise ValueError(
            f"unsupported BOT_PROFILE={profile!r}; expected one of {sorted(SUPPORTED_PROFILES)}"
        )
    return profile


def reverse_on_wait_sl() -> bool:
    return active_profile() == PROFILE_COUNTERTREND_SL_REVERSE


def strategy_instance_id() -> str:
    raw = str(os.getenv("BOT_INSTANCE_ID", "live_executor")).strip()
    return raw or "live_executor"


def display_name() -> str:
    """Human-facing label for this bot.

    Deliberately separate from BOT_PROFILE (which selects behaviour) and from
    BOT_INSTANCE_ID (which keys every row in Postgres). Renaming a bot should
    never change what it trades or orphan its history, so the friendly name
    lives in its own variable and can be changed freely.
    """
    raw = str(os.getenv("BOT_DISPLAY_NAME", "")).strip()
    return raw or strategy_instance_id()


def strategy_config_hash() -> str:
    profile = active_profile()
    instance = strategy_instance_id()
    return f"{instance}_{profile}_v1"


def _forced_countertrend_gate(profile: str) -> Dict[str, Any]:
    now = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "ts": now,
        "gate_on": 1,
        "gate_off": 0,
        "gate_countertrend_on": 1,
        "gate_trend_on": 0,
        "primary": "countertrend",
        "regime_state": "countertrend",
        "source": f"bot_profile:{profile}",
        "bot_profile": profile,
    }


def _rolling_rank_last(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("pc3axis_axis_has_no_finite_values")
    return float(clean.rank(method="average", pct=True).iloc[-1])


def _load_pc3axis_frame(path: Path) -> tuple[pd.DataFrame, str]:
    if path.exists():
        return pd.read_parquet(path), str(path)

    end_ts = pd.Timestamp.now("UTC")
    retention_days = max(1, int(os.getenv("PC3AXIS_RENKO_RETENTION_DAYS", "30")))
    symbol = str(os.getenv("LIVE_SYMBOL", "SOL-USDT")).strip().upper()
    renko = load_live_renko_bricks_from_postgres(
        symbol=symbol,
        start_ts=(end_ts - pd.Timedelta(days=retention_days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_ts=end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if renko is None or renko.empty:
        raise FileNotFoundError(f"pc3axis_state_space_missing:{path}; postgres_renko_empty:{symbol}")
    return compute_state_space(renko, StateSpaceConfig()), f"postgres_renko:{symbol}"


def _pc3axis_gate_from_state_space() -> Dict[str, Any]:
    path = Path(
        os.getenv(
            "PC3AXIS_STATE_SPACE_PATH",
            os.getenv("DASHBOARD_STATESPACE_PARQUET", "data/live/state_space_latest.parquet"),
        )
    )
    frame, input_source = _load_pc3axis_frame(path)
    required = {"ts", "X_raw", "Y_res", "Z_res"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pc3axis_state_space_missing_columns:{sorted(missing)}")

    frame = frame[list(required)].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    lookback = max(50, int(os.getenv("PC3AXIS_LOOKBACK_ROWS", "4000")))
    frame = frame.tail(lookback).reset_index(drop=True)
    if len(frame) < 50:
        raise ValueError(f"pc3axis_state_space_too_short:{len(frame)}")

    # Live adaptation of the last documented PC-gate winner:
    # instability <= q35, elasticity >= q25, |drift| <= q55, strict 3-of-3.
    drift_rank = _rolling_rank_last(frame["X_raw"].abs())
    elasticity_rank = _rolling_rank_last(frame["Y_res"].abs())
    instability_rank = _rolling_rank_last(frame["Z_res"].abs())

    drift_q = float(os.getenv("PC3AXIS_DRIFT_ABS_Q", "0.55"))
    elasticity_q = float(os.getenv("PC3AXIS_ELASTICITY_Q", "0.25"))
    instability_q = float(os.getenv("PC3AXIS_INSTABILITY_Q", "0.35"))

    drift_ok = int(drift_rank <= drift_q)
    elasticity_ok = int(elasticity_rank >= elasticity_q)
    instability_ok = int(instability_rank <= instability_q)
    gate_countertrend_on = int(drift_ok and elasticity_ok and instability_ok)
    ts = pd.Timestamp(frame.iloc[-1]["ts"]).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "ts": ts,
        "gate_on": gate_countertrend_on,
        "gate_off": int(1 - gate_countertrend_on),
        "gate_countertrend_on": gate_countertrend_on,
        "gate_trend_on": int(1 - gate_countertrend_on),
        "primary": "countertrend",
        "regime_state": "countertrend" if gate_countertrend_on else "trendfollower",
        "source": "bot_profile:pc3axis_state_space",
        "bot_profile": PROFILE_PC3AXIS,
        "pc3axis": {
            "mode": "strict_3of3",
            "input_source": input_source,
            "rows": int(len(frame)),
            "drift_abs_rank": drift_rank,
            "elasticity_abs_rank": elasticity_rank,
            "instability_abs_rank": instability_rank,
            "drift_abs_q": drift_q,
            "elasticity_q": elasticity_q,
            "instability_q": instability_q,
            "drift_ok": drift_ok,
            "elasticity_ok": elasticity_ok,
            "instability_ok": instability_ok,
        },
    }


def resolve_profile_gate(base_gate: Dict[str, Any] | None = None) -> Dict[str, Any]:
    profile = active_profile()
    if profile == PROFILE_CANONICAL:
        return dict(base_gate or {})
    if profile in {PROFILE_COUNTERTREND, PROFILE_COUNTERTREND_SL_REVERSE}:
        return _forced_countertrend_gate(profile)
    try:
        return _pc3axis_gate_from_state_space()
    except Exception as exc:
        out = dict(base_gate or {})
        out["bot_profile"] = profile
        out["profile_gate_error"] = str(exc)
        return out
