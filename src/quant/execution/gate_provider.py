from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class GateState:
    ts: str
    gate_on: int
    gate_off: int


def _live_default(rel_path: str) -> str:
    if Path("/data").exists():
        return str(Path("/data") / rel_path)
    return str(Path("data") / rel_path)


def _rank01(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    m = np.isfinite(a)
    out = np.full_like(a, np.nan, dtype=float)
    vals = a[m]
    if len(vals) == 0:
        return out
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(vals), dtype=float)
    out[m] = ranks / max(1.0, (len(vals) - 1))
    return out


def _rolling_slope_r2(x: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    if n < win or win < 2:
        return slope, r2

    t = np.arange(win, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    for i in range(win - 1, n):
        y = x[i - win + 1 : i + 1]
        y_mean = y.mean()
        cov = ((t - t_mean) * (y - y_mean)).sum()
        b = cov / t_var if t_var > 0 else 0.0
        a = y_mean - b * t_mean
        yhat = a + b * t
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        slope[i] = b
        r2[i] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return slope, r2


def _q_train(a: np.ndarray, train_slice: slice, q: float) -> float:
    x = np.asarray(a, dtype=float)[train_slice]
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float("nan")


def get_live_gate_from_statespace() -> Optional[Dict[str, Any]]:
    """
    Compute gate_on/gate_off from the live state space parquet.
    Returns None if state space data is unavailable or too old.

    3-axis gate logic (2-of-3):
      g1: |X_raw| <= drift_thresh     (not trending strongly)
      g2: Y_res >= elasticity_thresh  (mean-reverting)
      g3: Z_res <= instability_thresh (stable regime)
      gate_on = (g1 + g2 + g3) >= 2
    """
    ss_path = Path(os.getenv(
        "DASHBOARD_STATESPACE_PARQUET",
        _live_default("live/state_space_latest.parquet"),
    ))
    if not ss_path.exists():
        return None

    try:
        df = pd.read_parquet(ss_path)
    except Exception:
        return None

    need = {"ts", "X_raw", "Y_res", "Z_res"}
    if not need.issubset(set(df.columns)) or df.empty:
        return None

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "X_raw", "Y_res", "Z_res"]).sort_values("ts")
    if df.empty:
        return None

    max_age_sec = float(os.getenv("LIVE_GATE_MAX_AGE_SEC", "1800"))
    last_ts = df["ts"].iloc[-1]
    age = (pd.Timestamp.now("UTC") - last_ts).total_seconds()
    if age > max_age_sec:
        return None

    last = df.iloc[-1]
    x = float(last["X_raw"])
    y = float(last["Y_res"])
    z = float(last["Z_res"])

    drift_thresh = float(os.getenv("LIVE_GATE_DRIFT_THRESH", "0.3"))
    elasticity_thresh = float(os.getenv("LIVE_GATE_ELASTICITY_THRESH", "0.0"))
    instability_thresh = float(os.getenv("LIVE_GATE_INSTABILITY_THRESH", "0.3"))

    g1 = int(abs(x) <= drift_thresh)
    g2 = int(y >= elasticity_thresh)
    g3 = int(z <= instability_thresh)
    gate_on = int((g1 + g2 + g3) >= 2)

    ts_str = pd.Timestamp(last_ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "ts": ts_str,
        "gate_on": gate_on,
        "gate_off": int(1 - gate_on),
        "source": "statespace_live",
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z, 4),
        "g1_drift": g1,
        "g2_elasticity": g2,
        "g3_instability": g3,
        "age_sec": round(age, 1),
    }


def get_live_gate_state() -> Dict[str, Any]:
    """
    Build live gate state for SOL-USD from the existing PC gate definition logic.

    Priority (unless GATE_PREFER_CSV=1 for exact backtest parity):
      1. Live state space parquet (refreshed every 5 min)
      2. Predictions parquet (same logic as make_pc_3axis_gate_v2)
      3. Gate CSV (gate_base_2of3, exact backtest source)
      4. default_off
    """
    # GATE_PREFER_CSV=1 (default): use gate CSV first for exact backtest parity (gate_base_2of3)
    prefer_csv = str(os.getenv("GATE_PREFER_CSV", "1")).strip().lower() in ("1", "true", "yes", "on")
    if prefer_csv:
        gate_csv = Path(os.getenv("PC_GATE_CSV", _live_default("regimes/pc_3axis_gate_latest.csv")))
        if gate_csv.exists():
            try:
                df = pd.read_csv(gate_csv)
                if not df.empty and "gate_base_2of3" in df.columns:
                    row = df.iloc[-1]
                    gate_on = int(row.get("gate_base_2of3", 0) or 0)
                    ts = str(row.get("ts") or pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
                    return {"ts": ts, "gate_on": gate_on, "gate_off": int(1 - gate_on), "source": "gate_csv"}
            except Exception:
                pass

    ss_gate = get_live_gate_from_statespace()
    if ss_gate is not None:
        return ss_gate

    pred_path = Path(os.getenv("PC_PREDICTIONS_PARQUET", _live_default("runs/PC_GATE_FULLRANGE/pc_v02/predictions.parquet")))
    if not pred_path.exists():
        # Fallback to latest row from prebuilt gate CSV.
        gate_csv = Path(os.getenv("PC_GATE_CSV", _live_default("regimes/pc_3axis_gate_FULLRANGE_nolookahead_v2.csv")))
        if gate_csv.exists():
            df = pd.read_csv(gate_csv)
            if not df.empty and "gate_base_2of3" in df.columns:
                row = df.iloc[-1]
                gate_on = int(row.get("gate_base_2of3", 0) or 0)
                ts = str(row.get("ts") or pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
                return {"ts": ts, "gate_on": gate_on, "gate_off": int(1 - gate_on), "source": "gate_csv"}
        return {
            "ts": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gate_on": 0,
            "gate_off": 1,
            "source": "default_off",
            "error": f"missing_predictions:{pred_path}",
        }

    df = pd.read_parquet(pred_path)
    if df.empty or "ts" not in df.columns:
        return {
            "ts": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gate_on": 0,
            "gate_off": 1,
            "source": "default_off",
            "error": "predictions_empty",
        }

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "close", "v_temporal"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        return {
            "ts": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gate_on": 0,
            "gate_off": 1,
            "source": "default_off",
            "error": "predictions_invalid",
        }

    drift_win = int(os.getenv("PC_GATE_DRIFT_WIN", "240"))
    elas_h = int(os.getenv("PC_GATE_ELAS_H", "15"))
    train_frac = float(os.getenv("PC_GATE_TRAIN_FRAC", "0.70"))

    close = pd.to_numeric(df["close"], errors="coerce").astype(float).to_numpy()
    vt = pd.to_numeric(df["v_temporal"], errors="coerce").astype(float).to_numpy()
    vobs = pd.to_numeric(df.get("v_obs_mean"), errors="coerce").astype(float).to_numpy() if "v_obs_mean" in df.columns else None

    logp = np.log(np.where(close > 0, close, np.nan))
    n = len(df)
    cut = max(1, min(n, int(n * train_frac)))
    train = slice(0, cut)

    instab = _rank01(vt)
    if vobs is not None and len(vobs) == len(vt):
        instab = 0.7 * instab + 0.3 * _rank01(vobs)

    slope, r2 = _rolling_slope_r2(logp, drift_win)
    drift_raw = slope * np.clip(r2, 0.0, 1.0)
    tr = drift_raw[train]
    m = np.isfinite(tr)
    mu = float(np.nanmean(tr[m])) if m.any() else 0.0
    sd = float(np.nanstd(tr[m]) + 1e-12) if m.any() else 1.0
    drift_z = (drift_raw - mu) / sd
    drift_eff = drift_z * (1.0 - np.nan_to_num(instab, nan=0.0))

    r_past = np.full(n, np.nan)
    if n > elas_h:
        r_past[elas_h:] = np.log(close[elas_h:] / close[:-elas_h])
    elas = _rank01(np.abs(r_past))

    t_instab_40 = _q_train(instab, train, 0.40)
    t_elas_30 = _q_train(elas, train, 0.30)
    t_drift_60 = _q_train(np.abs(drift_eff), train, 0.60)

    g1 = (instab <= t_instab_40).astype(int)
    g2 = (elas >= t_elas_30).astype(int)
    g3 = (np.abs(drift_eff) <= t_drift_60).astype(int)
    g2of3 = ((g1 + g2 + g3) >= 2).astype(int)

    gate_on = int(g2of3[-1]) if len(g2of3) else 0
    ts = pd.Timestamp(df.iloc[-1]["ts"]).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "ts": ts,
        "gate_on": gate_on,
        "gate_off": int(1 - gate_on),
        "source": "predictions_parquet_live",
    }
