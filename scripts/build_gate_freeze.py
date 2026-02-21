# scripts/build_gate_freeze.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def smooth_gate(g: pd.Series, dw: int) -> pd.Series:
    """
    Past-only smoothing:
    - require dw consecutive ON days to turn ON
    - require dw consecutive OFF days to turn OFF
    """
    g = g.astype(int).fillna(0)

    if dw <= 1:
        return g

    on_run = g.rolling(dw, min_periods=dw).sum() >= dw
    off_run = (1 - g).rolling(dw, min_periods=dw).sum() >= dw

    out = np.zeros(len(g), dtype=int)
    state = 0
    for i in range(len(g)):
        if state == 0:
            if bool(on_run.iloc[i]):
                state = 1
        else:
            if bool(off_run.iloc[i]):
                state = 0
        out[i] = state

    return pd.Series(out, index=g.index)


def build_gate_fixed_thresholds(
    daily: pd.DataFrame,
    chop_on: float,
    chop_off: float,
    adx_on: float,
    dw: int,
) -> pd.Series:
    """
    Fixed thresholds:
    - ON when CHOP <= chop_on AND ADX >= adx_on
    - OFF when CHOP > chop_off OR ADX < adx_on
    - optional dw smoothing (past-only)
    """
    out = np.zeros(len(daily), dtype=int)
    state = 0

    for i in range(len(daily)):
        chop_v = float(daily["CHOP"].iloc[i])
        adx_v = float(daily["ADX"].iloc[i])

        if state == 0:
            if (chop_v <= chop_on) and (adx_v >= adx_on):
                state = 1
        else:
            if (chop_v > chop_off) or (adx_v < adx_on):
                state = 0

        out[i] = state

    gate = pd.Series(out, index=daily.index).astype(int)
    gate = smooth_gate(gate, dw)
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-features", required=True, help="Parquet with daily features (index=ts)")
    ap.add_argument("--out-csv", required=True, help="Output CSV with fixed-threshold gate")
    ap.add_argument("--freeze-start", required=True, help="Holdout start UTC date, e.g. 2025-05-01")

    # threshold estimation from pre-holdout
    ap.add_argument("--chon-q", type=float, default=0.75)
    ap.add_argument("--choff-q", type=float, default=0.65)
    ap.add_argument("--adx-q", type=float, default=0.50)

    ap.add_argument("--dw", type=int, default=3)
    args = ap.parse_args()

    daily = pd.read_parquet(args.daily_features).copy()
    daily.index = pd.to_datetime(daily.index, utc=True)
    daily = daily.sort_index()

    daily["CHOP"] = pd.to_numeric(daily["CHOP"], errors="coerce")
    daily["ADX"] = pd.to_numeric(daily["ADX"], errors="coerce")
    daily = daily.dropna(subset=["CHOP", "ADX"])

    freeze_start = pd.to_datetime(args.freeze_start, utc=True)

    pre = daily.loc[: freeze_start - pd.Timedelta(days=1)]
    hold = daily.loc[freeze_start:]

    if len(pre) < 60:
        raise ValueError(f"Not enough pre-holdout rows to fit thresholds: pre rows={len(pre)}")
    if len(hold) < 10:
        raise ValueError(f"Holdout too small: hold rows={len(hold)}")

    chop_on = float(pre["CHOP"].quantile(args.chon_q))
    chop_off = float(pre["CHOP"].quantile(args.choff_q))
    adx_on = float(pre["ADX"].quantile(args.adx_q))

    gate_hold = build_gate_fixed_thresholds(
        daily=hold,
        chop_on=chop_on,
        chop_off=chop_off,
        adx_on=adx_on,
        dw=args.dw,
    )

    out_df = pd.DataFrame(
        {
            "ts": gate_hold.index,
            "gate_on_freeze": gate_hold.values,
        }
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    meta = {
        "freeze_start": str(freeze_start),
        "pre_rows": int(len(pre)),
        "hold_rows": int(len(hold)),
        "chon_q": args.chon_q,
        "choff_q": args.choff_q,
        "adx_q": args.adx_q,
        "dw": args.dw,
        "chop_on": chop_on,
        "chop_off": chop_off,
        "adx_on": adx_on,
        "hold_on_rate": float(gate_hold.mean()),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(pd.Series(meta).to_json(indent=2), encoding="utf-8")

    print("wrote", out_path, "rows", len(out_df))
    print("wrote", meta_path)
    print("HOLD on-rate:", meta["hold_on_rate"])
    print("thresholds:", "chop_on", chop_on, "chop_off", chop_off, "adx_on", adx_on)


if __name__ == "__main__":
    main()
