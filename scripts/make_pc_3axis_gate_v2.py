# scripts/make_pc_3axis_gate_v2.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


PRED_PATH = Path("data/runs/PC_GATE_FULLRANGE/pc_v02/predictions.parquet")
OUT_PATH = Path("data/regimes/pc_3axis_gate_FULLRANGE_nolookahead_v2.csv")

DRIFT_WIN = 240   # ~4h on 1m bars
ELAS_H = 15       # past |r_15| proxy (causal)
TRAIN_FRAC = 0.70


def rank01(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    m = np.isfinite(a)
    out = np.full_like(a, np.nan, dtype=float)
    vals = a[m]
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(vals), dtype=float)
    out[m] = ranks / max(1.0, (len(vals) - 1))
    return out


def rolling_slope_r2(x: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)

    t = np.arange(win, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    for i in range(win - 1, n):
        y = x[i - win + 1 : i + 1]
        y_mean = y.mean()
        cov = ((t - t_mean) * (y - y_mean)).sum()
        b = cov / t_var
        a = y_mean - b * t_mean
        yhat = a + b * t
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        slope[i] = b
        r2[i] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return slope, r2


def q_train(a: np.ndarray, train_slice: slice, q: float) -> float:
    x = np.asarray(a, dtype=float)[train_slice]
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float("nan")


def gates(instab: np.ndarray, elas: np.ndarray, drift_eff: np.ndarray,
          t_instab: float, t_elas: float, t_drift_abs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g1 = (instab <= t_instab).astype(int)
    g2 = (elas >= t_elas).astype(int)
    g3 = (np.abs(drift_eff) <= t_drift_abs).astype(int)
    g2of3 = ((g1 + g2 + g3) >= 2).astype(int)
    g3of3 = ((g1 + g2 + g3) >= 3).astype(int)
    return g1, g2, g3, g2of3, g3of3


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(PRED_PATH)

    df = pd.read_parquet(PRED_PATH)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    close = df["close"].astype(float).to_numpy()
    logp = np.log(close)

    n = len(df)
    cut = int(n * TRAIN_FRAC)
    train = slice(0, cut)

    # --- Instability (causal): rank(v_temporal) [+ optional rank(v_obs_mean)] ---
    vt = df["v_temporal"].astype(float).to_numpy()
    vobs = df["v_obs_mean"].astype(float).to_numpy() if "v_obs_mean" in df.columns else None
    instab = rank01(vt)
    if vobs is not None:
        instab = 0.7 * instab + 0.3 * rank01(vobs)

    # --- Drift (causal): slope(logp)*R2; z-scored on TRAIN only; attenuated by (1-instab) ---
    slope, r2 = rolling_slope_r2(logp, DRIFT_WIN)
    drift_raw = slope * np.clip(r2, 0.0, 1.0)
    train_raw = drift_raw[train]
    m = np.isfinite(train_raw)
    mu = float(np.nanmean(train_raw[m]))
    sd = float(np.nanstd(train_raw[m]) + 1e-12)
    drift_z = (drift_raw - mu) / sd
    drift_eff = drift_z * (1.0 - np.nan_to_num(instab, nan=0.0))

    # --- Elasticity proxy (causal): past |r_15| rank ---
    r_past = np.full(n, np.nan)
    r_past[ELAS_H:] = np.log(close[ELAS_H:] / close[:-ELAS_H])
    elas = rank01(np.abs(r_past))

    # --- Threshold sets (TRAIN only) ---
    # Base: instab q40, elas q30, |drift| q60
    t_instab_40 = q_train(instab, train, 0.40)
    t_elas_30 = q_train(elas, train, 0.30)
    t_drift_60 = q_train(np.abs(drift_eff), train, 0.60)

    # Loose: instab q50, elas q20, |drift| q70
    t_instab_50 = q_train(instab, train, 0.50)
    t_elas_20 = q_train(elas, train, 0.20)
    t_drift_70 = q_train(np.abs(drift_eff), train, 0.70)

    g1, g2, g3, base_2of3, base_3of3 = gates(instab, elas, drift_eff, t_instab_40, t_elas_30, t_drift_60)
    _, _, _, loose_2of3, loose_3of3 = gates(instab, elas, drift_eff, t_instab_50, t_elas_20, t_drift_70)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "ts": df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate_base_instab_q40": g1,
        "gate_base_elas_q30": g2,
        "gate_base_drift_abs_q60": g3,
        "gate_base_2of3": base_2of3,
        "gate_base_3of3": base_3of3,
        "gate_loose_2of3": loose_2of3,
        "gate_loose_3of3": loose_3of3,
    })
    out.to_csv(OUT_PATH, index=False)

    print("WROTE:", OUT_PATH)
    print("TRAIN rows:", cut, "of", n, "CUT ts:", df["ts"].iloc[cut])
    print("thresholds_base:", {"instab_q40": t_instab_40, "elas_q30": t_elas_30, "drift_abs_q60": t_drift_60})
    print("thresholds_loose:", {"instab_q50": t_instab_50, "elas_q20": t_elas_20, "drift_abs_q70": t_drift_70})
    print("ON-rate base_2of3/base_3of3/loose_2of3/loose_3of3:",
          float(out["gate_base_2of3"].mean()),
          float(out["gate_base_3of3"].mean()),
          float(out["gate_loose_2of3"].mean()),
          float(out["gate_loose_3of3"].mean()))


if __name__ == "__main__":
    main()