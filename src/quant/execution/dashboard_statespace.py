from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from quant.state_space.config import StateSpaceConfig
from quant.state_space.pipeline import compute_state_space
from quant.utils.log import get_logger

log = get_logger("quant.dashboard_statespace")

_KEEP_COLS = ["ts", "X_raw", "Y_res", "Z_res", "conf_x", "conf_y", "conf_z"]


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _read_renko_df() -> pd.DataFrame:
    p = _env_path("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet")
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
    except Exception:
        log.warning("failed to read %s", p, exc_info=True)
        return pd.DataFrame()
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            return pd.DataFrame()
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def _read_state_space_df() -> pd.DataFrame:
    p = _env_path("DASHBOARD_STATESPACE_PARQUET", "data/live/state_space_latest.parquet")
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
    except Exception:
        log.warning("failed to read %s", p, exc_info=True)
        return pd.DataFrame()
    if "ts" not in df.columns:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def refresh_state_space_cache() -> Dict[str, Any]:
    """Compute state space from renko data and write to parquet cache."""
    renko = _read_renko_df()
    if renko.empty:
        return {"ok": False, "reason": "renko_file_missing_or_empty"}
    if len(renko) < 50:
        return {"ok": False, "reason": f"renko_too_short ({len(renko)} rows)"}

    ss = compute_state_space(renko, StateSpaceConfig())

    keep = [c for c in _KEEP_COLS if c in ss.columns]
    out = ss[keep].copy()
    out = out.dropna(subset=["X_raw", "Y_res", "Z_res"])

    out_path = _env_path("DASHBOARD_STATESPACE_PARQUET", "data/live/state_space_latest.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    return {"ok": True, "rows": len(out), "path": str(out_path)}


def load_state_space_trajectory(window_hours: float = 8.0) -> Dict[str, Any]:
    """Load state space trajectory filtered by time window."""
    df = _read_state_space_df()
    if df.empty:
        return {"trajectory": [], "current": None}

    max_ts = df["ts"].max()
    cutoff = max_ts - pd.Timedelta(hours=window_hours)
    df = df[df["ts"] >= cutoff].reset_index(drop=True)

    if df.empty:
        return {"trajectory": [], "current": None}

    trajectory: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        trajectory.append({
            "ts": int(pd.Timestamp(row["ts"]).timestamp()),
            "x": float(row["X_raw"]),
            "y": float(row["Y_res"]),
            "z": float(row["Z_res"]),
        })

    last = df.iloc[-1]
    current = {
        "x": float(last["X_raw"]),
        "y": float(last["Y_res"]),
        "z": float(last["Z_res"]),
        "conf_x": float(last.get("conf_x", 0.0)),
        "conf_y": float(last.get("conf_y", 0.0)),
        "conf_z": float(last.get("conf_z", 0.0)),
    }

    return {"trajectory": trajectory, "current": current}
