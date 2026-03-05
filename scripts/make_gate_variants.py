#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariantSpec:
    name: str
    kind: str  # "debounce" | "debounce_hold" | "majority"
    confirm: int = 0
    hold: int = 0
    window: int = 0


def _as_int01(s: pd.Series, default_off: int = 0) -> np.ndarray:
    # Accept {0,1}, bool, floats with NaN.
    x = s.copy()
    x = x.fillna(default_off)
    x = x.astype(float)
    x = (x > 0.5).astype(np.int8)
    return x.to_numpy(dtype=np.int8, copy=False)


def apply_debounce(g: np.ndarray, confirm: int) -> np.ndarray:
    """
    Debounce / grace-before-flip:
    Switch only after `confirm` consecutive samples of the new state.
    """
    n = int(g.shape[0])
    out = np.empty(n, dtype=np.int8)
    state = int(g[0])
    pending = state
    cnt = 0

    for i in range(n):
        gi = int(g[i])
        if gi == state:
            pending = state
            cnt = 0
        else:
            if gi != pending:
                pending = gi
                cnt = 1
            else:
                cnt += 1
            if cnt >= confirm:
                state = pending
                pending = state
                cnt = 0
        out[i] = state
    return out


def apply_debounce_with_hold(g: np.ndarray, confirm: int, hold: int) -> np.ndarray:
    """
    Debounce + minimum hold time after a switch.
    Hold is enforced in samples (rows).
    """
    n = int(g.shape[0])
    out = np.empty(n, dtype=np.int8)
    state = int(g[0])
    pending = state
    cnt = 0
    hold_left = 0

    for i in range(n):
        gi = int(g[i])

        if hold_left > 0:
            # during hold, ignore changes
            hold_left -= 1
            pending = state
            cnt = 0
            out[i] = state
            continue

        if gi == state:
            pending = state
            cnt = 0
        else:
            if gi != pending:
                pending = gi
                cnt = 1
            else:
                cnt += 1
            if cnt >= confirm:
                state = pending
                pending = state
                cnt = 0
                hold_left = max(int(hold), 0)
        out[i] = state

    return out


def apply_majority(g: np.ndarray, window: int) -> np.ndarray:
    """
    Trailing rolling majority over the last `window` samples.
    (window=1 -> identity)
    """
    w = max(int(window), 1)
    if w == 1:
        return g.copy()

    s = pd.Series(g.astype(np.int8))
    m = s.rolling(w, min_periods=w).mean()
    out = np.where(m.isna(), s, (m >= 0.5).astype(np.int8)).astype(np.int8)
    return out.to_numpy(dtype=np.int8, copy=False)


def summarize_gate(name: str, g: np.ndarray) -> Dict[str, float]:
    flips = int(np.sum(g[1:] != g[:-1]))
    on_rate = float(np.mean(g))
    # run lengths
    # boundaries where value changes
    idx = np.flatnonzero(g[1:] != g[:-1]) + 1
    starts = np.r_[0, idx]
    ends = np.r_[idx, len(g)]
    lens = (ends - starts).astype(int)
    med_len = float(np.median(lens)) if len(lens) else float("nan")
    p90_len = float(np.percentile(lens, 90)) if len(lens) else float("nan")
    return {
        "on_rate": on_rate,
        "flips": flips,
        "n_segments": float(len(lens)),
        "median_seg_len": med_len,
        "p90_seg_len": p90_len,
    }


def parse_variants(s: str) -> List[VariantSpec]:
    """
    Format:
      debounce:K
      debounce_hold:K:H
      majority:W
    Comma-separated list.
    Example:
      "debounce:3,debounce:5,debounce_hold:3:10,majority:11"
    """
    specs: List[VariantSpec] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        items = p.split(":")
        kind = items[0].strip()
        if kind == "debounce":
            k = int(items[1])
            specs.append(VariantSpec(name=f"debounce{k}", kind="debounce", confirm=k))
        elif kind == "debounce_hold":
            k = int(items[1])
            h = int(items[2])
            specs.append(
                VariantSpec(name=f"debounce{k}_hold{h}", kind="debounce_hold", confirm=k, hold=h)
            )
        elif kind == "majority":
            w = int(items[1])
            specs.append(VariantSpec(name=f"maj{w}", kind="majority", window=w))
        else:
            raise ValueError(f"Unknown variant kind: {kind} (in '{p}')")
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Create smoothed/hysteresis variants of a binary gate column.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV (must include ts + gate column).")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path.")
    ap.add_argument("--ts-col", default="ts", help="Timestamp column name (default: ts).")
    ap.add_argument("--gate-col", required=True, help="Binary gate column to transform (0/1 or bool).")
    ap.add_argument("--default-off", type=int, default=0, help="Fill NaNs with this value (default: 0).")
    ap.add_argument(
        "--variants",
        default="debounce:3,debounce:5,debounce_hold:3:10,debounce_hold:5:10,majority:11",
        help="Comma list: debounce:K | debounce_hold:K:H | majority:W",
    )
    ap.add_argument(
        "--minimal",
        action="store_true",
        help="Write only ts + original gate + variant columns (recommended for huge files).",
    )
    ap.add_argument(
        "--invert",
        action="store_true",
        help="Also write inverted versions of each variant (suffix _inv).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.ts_col not in df.columns:
        raise SystemExit(f"Missing ts column '{args.ts_col}' in {in_path}")
    if args.gate_col not in df.columns:
        raise SystemExit(f"Missing gate column '{args.gate_col}' in {in_path}")

    # Parse timestamps (keep as ISO in output)
    ts = pd.to_datetime(df[args.ts_col], utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise SystemExit(f"Found {bad} unparsable timestamps in column '{args.ts_col}'")
    df[args.ts_col] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    g0 = _as_int01(df[args.gate_col], default_off=args.default_off)

    specs = parse_variants(args.variants)

    out_cols: Dict[str, np.ndarray] = {}
    out_cols[args.gate_col] = g0

    print("\n=== Gate variant summary (on_rate, flips, segment stats) ===")
    base_stats = summarize_gate("base", g0)
    print(f"base           on_rate={base_stats['on_rate']:.4f} flips={int(base_stats['flips'])} "
          f"med_len={base_stats['median_seg_len']:.1f} p90_len={base_stats['p90_seg_len']:.1f}")

    for spec in specs:
        if spec.kind == "debounce":
            gv = apply_debounce(g0, confirm=spec.confirm)
        elif spec.kind == "debounce_hold":
            gv = apply_debounce_with_hold(g0, confirm=spec.confirm, hold=spec.hold)
        elif spec.kind == "majority":
            gv = apply_majority(g0, window=spec.window)
        else:
            raise SystemExit(f"Unhandled kind {spec.kind}")

        out_cols[f"{args.gate_col}_{spec.name}"] = gv
        st = summarize_gate(spec.name, gv)
        print(f"{spec.name:<14} on_rate={st['on_rate']:.4f} flips={int(st['flips'])} "
              f"med_len={st['median_seg_len']:.1f} p90_len={st['p90_seg_len']:.1f}")

        if args.invert:
            inv = (1 - gv).astype(np.int8)
            out_cols[f"{args.gate_col}_{spec.name}_inv"] = inv
            st2 = summarize_gate(spec.name + "_inv", inv)
            print(f"{spec.name+'_inv':<14} on_rate={st2['on_rate']:.4f} flips={int(st2['flips'])} "
                  f"med_len={st2['median_seg_len']:.1f} p90_len={st2['p90_seg_len']:.1f}")

    if args.minimal:
        out_df = pd.DataFrame({args.ts_col: df[args.ts_col]})
        # keep original gate col + variants
        for k, v in out_cols.items():
            out_df[k] = v
    else:
        out_df = df.copy()
        for k, v in out_cols.items():
            out_df[k] = v

    out_df.to_csv(out_path, index=False)
    print(f"\nWROTE: {out_path}  rows={len(out_df)}  cols={len(out_df.columns)}\n")


if __name__ == "__main__":
    main()