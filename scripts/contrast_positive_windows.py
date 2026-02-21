# scripts/contrast_positive_windows.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp


def parse_positive_windows() -> List[Window]:
    raw: List[Tuple[str, str]] = [
        ("2024-01-24", "2024-02-13"),
        ("2024-02-27", "2024-03-15"),
        ("2024-03-26", "2024-04-12"),
        ("2024-05-28", "2024-07-03"),
        ("2024-10-25", "2024-12-10"),
        ("2024-12-24", "2025-01-02"),
        ("2025-09-16", "2025-10-23"),
        ("2025-11-07", "2025-12-12"),
    ]
    return [Window(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for a, b in raw]


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: missing columns: {missing}. Columns are: {list(df.columns)}")


def overlaps(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    # overlap if max(starts) < min(ends)
    return max(a_start, b_start) < min(a_end, b_end)


def mark_positive_windows_by_start_end(df: pd.DataFrame, pos_windows: List[Window]) -> pd.DataFrame:
    out = df.copy()

    _require_cols(out, ["start", "end"])
    out["start"] = pd.to_datetime(out["start"], utc=True, errors="coerce")
    out["end"] = pd.to_datetime(out["end"], utc=True, errors="coerce")
    out = out.dropna(subset=["start", "end"]).reset_index(drop=True)

    is_pos = []
    for _, r in out.iterrows():
        s = r["start"]
        e = r["end"]
        hit = any(overlaps(s, e, w.start, w.end) for w in pos_windows)
        is_pos.append(hit)

    out["is_positive_window"] = np.array(is_pos, dtype=bool)
    return out


def auc_score(y_true: np.ndarray, x: np.ndarray) -> float:
    y = y_true.astype(int)
    mask = np.isfinite(x)
    y = y[mask]
    x = x[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    ranks = pd.Series(x).rank(method="average").values
    n1 = y.sum()
    n0 = len(y) - n1
    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def summarize_feature(df: pd.DataFrame, col: str) -> dict:
    x = pd.to_numeric(df[col], errors="coerce").astype(float)
    y = df["is_positive_window"].astype(int).values

    x_pos = x[df["is_positive_window"]].dropna()
    x_rest = x[~df["is_positive_window"]].dropna()

    auc_hi = auc_score(y_true=y, x=x.values)
    auc_lo = 1.0 - auc_hi if np.isfinite(auc_hi) else float("nan")

    def q(s: pd.Series, p: float) -> float:
        return float(s.quantile(p)) if len(s) else float("nan")

    return {
        "feature": col,
        "auc_high_is_positive": auc_hi,
        "auc_low_is_positive": auc_lo,
        "pos_n": int(len(x_pos)),
        "rest_n": int(len(x_rest)),
        "pos_mean": float(x_pos.mean()) if len(x_pos) else float("nan"),
        "rest_mean": float(x_rest.mean()) if len(x_rest) else float("nan"),
        "pos_p25": q(x_pos, 0.25),
        "pos_p50": q(x_pos, 0.50),
        "pos_p75": q(x_pos, 0.75),
        "rest_p25": q(x_rest, 0.25),
        "rest_p50": q(x_rest, 0.50),
        "rest_p75": q(x_rest, 0.75),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features)

    # This script expects window-level rows with start/end
    if ("start" not in df.columns) or ("end" not in df.columns):
        raise SystemExit(
            "ERROR: features CSV has no 'start'/'end'. "
            "This version is for window-aggregated features. "
            f"Columns are: {list(df.columns)}"
        )

    pos_windows = parse_positive_windows()
    df = mark_positive_windows_by_start_end(df, pos_windows)

    # pick numeric columns to score
    exclude = {"start", "end", "is_positive_window"}
    cols = [c for c in df.columns if c not in exclude]

    numeric_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if float(s.notna().mean()) > 0.80:
            numeric_cols.append(c)

    rows = [summarize_feature(df, c) for c in numeric_cols]
    out = pd.DataFrame(rows)

    out["best_auc"] = out[["auc_high_is_positive", "auc_low_is_positive"]].max(axis=1)
    out["auc_edge"] = (out["best_auc"] - 0.5).abs()
    out = out.sort_values("auc_edge", ascending=False).reset_index(drop=True)

    out.to_csv(args.out, index=False)

    print("INFO rows=", len(df), " positives=", int(df["is_positive_window"].sum()))
    print("WROTE", args.out)
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
