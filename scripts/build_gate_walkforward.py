# scripts/build_gate_walkforward.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # inclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp       # inclusive


def make_folds(idx: pd.DatetimeIndex, train_days: int, test_days: int, step_days: int) -> List[Fold]:
    idx = pd.DatetimeIndex(sorted(idx.unique()))
    start = idx.min().normalize()
    end = idx.max().normalize()

    folds: List[Fold] = []
    t0 = start + pd.Timedelta(days=train_days)
    while True:
        test_start = t0
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=train_days - 1)

        if test_end > end:
            break

        folds.append(Fold(train_start, train_end, test_start, test_end))
        t0 = t0 + pd.Timedelta(days=step_days)

    return folds


def smooth_gate(g: pd.Series, dw: int) -> pd.Series:
    g = g.astype(int).fillna(0)

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


def build_gate_for_period(
    daily_test: pd.DataFrame,
    train: pd.DataFrame,
    chon_q: float,
    choff_q: float,
    adx_q: float,
    dw: int,
) -> Tuple[pd.Series, dict]:
    """
    Online gate (no future):
    - Fit thresholds on TRAIN only.
    - Gate ON when CHOP is LOW (<= chop_on) AND ADX is HIGH (>= adx_on).
    - Hysteresis: stay ON until CHOP rises above chop_off OR ADX falls below adx_on.
    """
    chop_on = float(train["CHOP"].quantile(chon_q))
    chop_off = float(train["CHOP"].quantile(choff_q))
    adx_on = float(train["ADX"].quantile(adx_q))

    out = np.zeros(len(daily_test), dtype=int)
    state = 0
    for i in range(len(daily_test)):
        chop_v = float(daily_test["CHOP"].iloc[i])
        adx_v = float(daily_test["ADX"].iloc[i])

        if state == 0:
            if (chop_v <= chop_on) and (adx_v >= adx_on):
                state = 1
        else:
            if (chop_v > chop_off) or (adx_v < adx_on):
                state = 0

        out[i] = state

    gate = pd.Series(out, index=daily_test.index).astype(int)
    if dw and dw > 1:
        gate = smooth_gate(gate, dw)

    meta = {"chop_on": chop_on, "chop_off": chop_off, "adx_on": adx_on}
    return gate, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-features", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--train-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--chon-q", type=float, default=0.75)
    ap.add_argument("--choff-q", type=float, default=0.65)
    ap.add_argument("--adx-q", type=float, default=0.50)
    ap.add_argument("--dw", type=int, default=3)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    daily = pd.read_parquet(args.daily_features).copy()
    daily.index = pd.to_datetime(daily.index, utc=True)
    daily = daily.sort_index()

    if args.start:
        daily = daily.loc[pd.to_datetime(args.start, utc=True) :]
    if args.end:
        daily = daily.loc[: pd.to_datetime(args.end, utc=True)]

    daily["CHOP"] = pd.to_numeric(daily["CHOP"], errors="coerce")
    daily["ADX"] = pd.to_numeric(daily["ADX"], errors="coerce")
    daily = daily.dropna(subset=["CHOP", "ADX"])

    folds = make_folds(daily.index, args.train_days, args.test_days, args.step_days)
    if not folds:
        raise ValueError("No folds produced. Check date range vs train/test/step days.")

    out_gate = pd.Series(index=daily.index, dtype=float)
    out_fold = pd.Series(index=daily.index, dtype=float)
    meta_rows = []

    for k, f in enumerate(folds):
        train = daily.loc[f.train_start : f.train_end]
        test = daily.loc[f.test_start : f.test_end]

        if len(train) < 30 or len(test) < 5:
            continue

        gate_test, meta = build_gate_for_period(
            daily_test=test,
            train=train,
            chon_q=args.chon_q,
            choff_q=args.choff_q,
            adx_q=args.adx_q,
            dw=args.dw,
        )

        out_gate.loc[test.index] = gate_test.values
        out_fold.loc[test.index] = k

        meta_rows.append(
            {
                "fold": k,
                "train_start": str(f.train_start),
                "train_end": str(f.train_end),
                "test_start": str(f.test_start),
                "test_end": str(f.test_end),
                **meta,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            }
        )

    out_df = pd.DataFrame(
        {
            "ts": out_gate.index,
            "gate_on_wf": out_gate.fillna(0).astype(int).values,
            "fold": out_fold.fillna(-1).astype(int).values,
        }
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    meta_path = out_path.with_suffix(".meta.csv")
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    print("wrote", out_path, "rows", len(out_df))
    print("wrote", meta_path, "rows", len(meta_rows))
    print("overall ON-rate:", float(out_df["gate_on_wf"].mean()))


if __name__ == "__main__":
    main()
