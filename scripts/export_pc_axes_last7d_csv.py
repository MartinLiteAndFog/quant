from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant.state_space.pipeline import compute_state_space


RENKO_PATH = Path("/Users/martinpeter/Desktop/quant/data/renko/SOL-USDT_renko_box0.03_last7d_binance_ns.parquet")
GATE_PATH = Path("/Users/martinpeter/Desktop/quant-main/data/regimes/pc_3axis_gate_FULLRANGE_nolookahead_v2.csv")
CSV_OUT_PATH = Path("/Users/martinpeter/Desktop/quant-main/data/runs/pc_axes_spectral_last7d.csv")

EMA_SPAN = 12
ON_THRESHOLD = 0.42
OFF_THRESHOLD = 0.30
MIN_ON_BARS = 6
MIN_OFF_BARS = 4

W_DRIFT = 1.00
W_ELASTICITY = 0.90
W_INSTABILITY = 1.15


def normalize_signed(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    vmax = np.nanmax(np.abs(v.values))
    if not np.isfinite(vmax) or vmax <= 1e-12:
        return pd.Series(np.zeros(len(v)), index=v.index, dtype=float)
    return v / vmax


def build_raw_score(df: pd.DataFrame) -> pd.Series:
    drift = pd.to_numeric(df["X_norm"], errors="coerce").fillna(0.0)
    elasticity = pd.to_numeric(df["Y_norm"], errors="coerce").fillna(0.0)
    instability = pd.to_numeric(df["Z_norm"], errors="coerce").fillna(0.0)

    # Favor:
    # - high absolute drift
    # - high absolute elasticity
    # - low absolute instability
    score = (
        W_DRIFT * drift.abs()
        + W_ELASTICITY * elasticity.abs()
        - W_INSTABILITY * instability.abs()
    )

    vmax = float(np.nanmax(np.abs(score.values))) if len(score) else 0.0
    if not np.isfinite(vmax) or vmax <= 1e-12:
        return pd.Series(np.zeros(len(score)), index=score.index, dtype=float)

    return (score / vmax).clip(-1.0, 1.0)


def apply_hysteresis_with_persistence(
    score_ema: pd.Series,
    on_threshold: float,
    off_threshold: float,
    min_on_bars: int,
    min_off_bars: int,
) -> pd.Series:
    score = pd.to_numeric(score_ema, errors="coerce").fillna(0.0).values

    gate = np.zeros(len(score), dtype=int)
    state = 0
    bars_since_switch = 10**9

    for i, s in enumerate(score):
        if state == 0:
            if bars_since_switch >= min_off_bars and s >= on_threshold:
                state = 1
                bars_since_switch = 0
            else:
                bars_since_switch += 1
        else:
            if bars_since_switch >= min_on_bars and s <= off_threshold:
                state = 0
                bars_since_switch = 0
            else:
                bars_since_switch += 1

        gate[i] = state

    return pd.Series(gate, index=score_ema.index, dtype=int)


def main() -> None:
    if not RENKO_PATH.exists():
        raise FileNotFoundError(f"Missing renko parquet: {RENKO_PATH}")
    if not GATE_PATH.exists():
        raise FileNotFoundError(f"Missing gate csv: {GATE_PATH}")

    df = pd.read_parquet(RENKO_PATH).copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["ts", "open", "high", "low", "close"])
        .sort_values("ts")
        .reset_index(drop=True)
    )

    ss = compute_state_space(df[["ts", "open", "high", "low", "close"]].copy()).copy()
    ss["ts"] = pd.to_datetime(ss["ts"], utc=True, errors="coerce")
    ss = ss.sort_values("ts").reset_index(drop=True)

    # Mapping for the React tool:
    # Drift       <- X_raw
    # Elasticity  <- Y_res
    # Instability <- Z_res
    ss["X_norm"] = normalize_signed(ss["X_raw"])
    ss["Y_norm"] = normalize_signed(ss["Y_res"])
    ss["Z_norm"] = normalize_signed(ss["Z_res"])

    gate = pd.read_csv(GATE_PATH).copy()
    gate["ts"] = pd.to_datetime(gate["ts"], utc=True, errors="coerce")
    gate = gate.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if "gate_base_3of3" not in gate.columns:
        raise ValueError("Expected gate_base_3of3 in gate csv")

    gate["gate_on"] = (
        pd.to_numeric(gate["gate_base_3of3"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    base = df[["ts", "close"]].copy().sort_values("ts").reset_index(drop=True)

    gate_aligned = pd.merge_asof(
        base[["ts"]].sort_values("ts"),
        gate[["ts", "gate_on"]].sort_values("ts"),
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    gate_aligned["gate_on"] = gate_aligned["gate_on"].fillna(0).astype(int)

    export_df = pd.merge_asof(
        base.sort_values("ts"),
        ss[["ts", "X_norm", "Y_norm", "Z_norm"]].sort_values("ts"),
        on="ts",
        direction="nearest",
        allow_exact_matches=True,
    )

    export_df["X_norm"] = export_df["X_norm"].fillna(0.0)
    export_df["Y_norm"] = export_df["Y_norm"].fillna(0.0)
    export_df["Z_norm"] = export_df["Z_norm"].fillna(0.0)
    export_df["gate_on"] = gate_aligned["gate_on"].values

    export_df["score_raw"] = build_raw_score(export_df)
    export_df["score_ema"] = (
        export_df["score_raw"]
        .ewm(span=EMA_SPAN, adjust=False, min_periods=1)
        .mean()
        .clip(-1.0, 1.0)
    )

    export_df["gate_derived"] = apply_hysteresis_with_persistence(
        export_df["score_ema"],
        on_threshold=ON_THRESHOLD,
        off_threshold=OFF_THRESHOLD,
        min_on_bars=MIN_ON_BARS,
        min_off_bars=MIN_OFF_BARS,
    )

    export_df["ts"] = pd.to_datetime(export_df["ts"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    CSV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_df[
        [
            "ts",
            "close",
            "X_norm",
            "Y_norm",
            "Z_norm",
            "score_raw",
            "score_ema",
            "gate_on",
            "gate_derived",
        ]
    ].to_csv(CSV_OUT_PATH, index=False)

    print(f"WROTE CSV: {CSV_OUT_PATH}")
    print(
        export_df[
            [
                "ts",
                "close",
                "X_norm",
                "Y_norm",
                "Z_norm",
                "score_raw",
                "score_ema",
                "gate_on",
                "gate_derived",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )
    print()
    print(
        {
            "ema_span": EMA_SPAN,
            "on_threshold": ON_THRESHOLD,
            "off_threshold": OFF_THRESHOLD,
            "min_on_bars": MIN_ON_BARS,
            "min_off_bars": MIN_OFF_BARS,
            "gate_on_rate_csv": float(export_df["gate_on"].mean()),
            "gate_on_rate_derived": float(export_df["gate_derived"].mean()),
        }
    )


if __name__ == "__main__":
    main()