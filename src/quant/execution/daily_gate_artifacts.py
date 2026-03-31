from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


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


def gates(
    instab: np.ndarray,
    elas: np.ndarray,
    drift_eff: np.ndarray,
    t_instab: float,
    t_elas: float,
    t_drift_abs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g1 = (instab <= t_instab).astype(int)
    g2 = (elas >= t_elas).astype(int)
    g3 = (np.abs(drift_eff) <= t_drift_abs).astype(int)
    g2of3 = ((g1 + g2 + g3) >= 2).astype(int)
    g3of3 = ((g1 + g2 + g3) >= 3).astype(int)
    return g1, g2, g3, g2of3, g3of3


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _compute_pc_gate_frame(
    df: pd.DataFrame,
    *,
    drift_win: int = 240,
    elas_h: int = 15,
    train_frac: float = 0.70,
) -> pd.DataFrame:
    work = df.copy()
    if "ts" not in work.columns:
        raise ValueError("missing_ts_column")
    need = {"close", "v_temporal"}
    missing = need - set(work.columns)
    if missing:
        raise ValueError(f"missing_prediction_columns:{sorted(missing)}")

    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if len(work) < max(int(drift_win) + 5, 100):
        raise ValueError("not_enough_prediction_rows")

    close = pd.to_numeric(work["close"], errors="coerce").astype(float).to_numpy()
    logp = np.log(close)

    n = len(work)
    cut = int(n * float(train_frac))
    if cut <= int(drift_win):
        raise ValueError("prediction_train_window_too_small")
    train = slice(0, cut)

    vt = pd.to_numeric(work["v_temporal"], errors="coerce").astype(float).to_numpy()
    vobs = (
        pd.to_numeric(work["v_obs_mean"], errors="coerce").astype(float).to_numpy()
        if "v_obs_mean" in work.columns
        else None
    )
    instab = rank01(vt)
    if vobs is not None:
        instab = 0.7 * instab + 0.3 * rank01(vobs)

    slope, r2 = rolling_slope_r2(logp, int(drift_win))
    drift_raw = slope * np.clip(r2, 0.0, 1.0)
    train_raw = drift_raw[train]
    m = np.isfinite(train_raw)
    if not np.any(m):
        raise ValueError("no_finite_train_drift")
    mu = float(np.nanmean(train_raw[m]))
    sd = float(np.nanstd(train_raw[m]) + 1e-12)
    drift_z = (drift_raw - mu) / sd
    drift_eff = drift_z * (1.0 - np.nan_to_num(instab, nan=0.0))

    r_past = np.full(n, np.nan)
    h = int(elas_h)
    r_past[h:] = np.log(close[h:] / close[:-h])
    elas = rank01(np.abs(r_past))

    t_instab_40 = q_train(instab, train, 0.40)
    t_elas_30 = q_train(elas, train, 0.30)
    t_drift_60 = q_train(np.abs(drift_eff), train, 0.60)

    t_instab_50 = q_train(instab, train, 0.50)
    t_elas_20 = q_train(elas, train, 0.20)
    t_drift_70 = q_train(np.abs(drift_eff), train, 0.70)

    g1, g2, g3, base_2of3, base_3of3 = gates(
        instab,
        elas,
        drift_eff,
        t_instab_40,
        t_elas_30,
        t_drift_60,
    )
    _, _, _, loose_2of3, loose_3of3 = gates(
        instab,
        elas,
        drift_eff,
        t_instab_50,
        t_elas_20,
        t_drift_70,
    )

    return pd.DataFrame(
        {
            "ts": work["ts"],
            "gate_base_instab_q40": g1,
            "gate_base_elas_q30": g2,
            "gate_base_drift_abs_q60": g3,
            "gate_base_2of3": base_2of3,
            "gate_base_3of3": base_3of3,
            "gate_loose_2of3": loose_2of3,
            "gate_loose_3of3": loose_3of3,
        }
    )


def build_daily_gate_artifacts(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    on_source_col: str = "gate_base_2of3",
    off_source_col: Optional[str] = None,
    on_output_col: str = "gate_on_2of3",
    off_output_col: str = "gate_off_2of3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if ts_col not in work.columns:
        raise ValueError(f"missing_ts_column:{ts_col}")
    if on_source_col not in work.columns:
        raise ValueError(f"missing_on_source_col:{on_source_col}")

    work[ts_col] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    work = work.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    work["day"] = work[ts_col].dt.floor("D")
    work[on_source_col] = pd.to_numeric(work[on_source_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    daily_on = work.groupby("day", as_index=False).tail(1).copy()
    daily_on = daily_on[["day", on_source_col]].rename(columns={"day": "ts", on_source_col: on_output_col})
    daily_on["ts"] = pd.to_datetime(daily_on["ts"], utc=True).dt.strftime("%Y-%m-%dT00:00:00Z")
    daily_on[on_output_col] = pd.to_numeric(daily_on[on_output_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    daily_on = daily_on.reset_index(drop=True)

    if off_source_col:
        if off_source_col not in work.columns:
            raise ValueError(f"missing_off_source_col:{off_source_col}")
        work[off_source_col] = pd.to_numeric(work[off_source_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        daily_off = work.groupby("day", as_index=False).tail(1).copy()
        daily_off = daily_off[["day", off_source_col]].rename(columns={"day": "ts", off_source_col: off_output_col})
        daily_off["ts"] = pd.to_datetime(daily_off["ts"], utc=True).dt.strftime("%Y-%m-%dT00:00:00Z")
        daily_off[off_output_col] = pd.to_numeric(daily_off[off_output_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        daily_off = daily_off.reset_index(drop=True)
    else:
        daily_off = pd.DataFrame(
            {
                "ts": daily_on["ts"],
                off_output_col: (1 - daily_on[on_output_col].astype(int)).astype(int),
            }
        )

    return daily_on, daily_off


def write_daily_gate_artifacts(
    *,
    input_path: str | Path,
    out_on_path: str | Path,
    out_off_path: str | Path,
    ts_col: str = "ts",
    on_source_col: str = "gate_base_2of3",
    off_source_col: Optional[str] = None,
    on_output_col: str = "gate_on_2of3",
    off_output_col: str = "gate_off_2of3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    src_path = Path(input_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    src_df = _load_frame(src_path)
    if on_source_col not in src_df.columns:
        src_df = _compute_pc_gate_frame(src_df)

    on_df, off_df = build_daily_gate_artifacts(
        src_df,
        ts_col=ts_col,
        on_source_col=on_source_col,
        off_source_col=off_source_col,
        on_output_col=on_output_col,
        off_output_col=off_output_col,
    )

    out_on = Path(out_on_path)
    out_off = Path(out_off_path)
    out_on.parent.mkdir(parents=True, exist_ok=True)
    out_off.parent.mkdir(parents=True, exist_ok=True)
    on_df.to_csv(out_on, index=False)
    off_df.to_csv(out_off, index=False)
    return on_df, off_df
