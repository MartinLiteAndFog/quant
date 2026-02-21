#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from quant.strategies.signal_io import load_signals


# -----------------------
# IO helpers
# -----------------------
def read_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    need = {"open", "high", "low", "close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing columns: {sorted(missing)}")

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def read_fills_parquet(path: str, fill_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("fills parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    if fill_col not in df.columns:
        raise ValueError(f"fills parquet missing fill_col='{fill_col}' (cols={list(df.columns)})")

    df[fill_col] = pd.to_numeric(df[fill_col], errors="coerce")
    df = df.dropna(subset=[fill_col]).reset_index(drop=True)
    return df[["ts", fill_col]].copy()


def signals_bundle_to_df(bundle) -> pd.DataFrame:
    if hasattr(bundle, "signal") and bundle.signal is not None:
        s = bundle.signal.copy()
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
        s = s[~s.index.isna()]
        s = s[~s.index.duplicated(keep="last")]
        vals = pd.to_numeric(pd.Series(s.values), errors="coerce").fillna(0).astype(int).values
        sig_df = pd.DataFrame({"ts": s.index, "signal": vals})
        sig_df = sig_df[sig_df["signal"] != 0].copy()
        return sig_df.sort_values("ts").reset_index(drop=True)

    if hasattr(bundle, "df") and bundle.df is not None:
        df = bundle.df.copy()
        if "ts" not in df.columns:
            raise ValueError("signals bundle.df missing 'ts'")

        if "signal" not in df.columns and "position" in df.columns:
            df = df.rename(columns={"position": "signal"})
        if "signal" not in df.columns and "action" in df.columns:

            def _map(a):
                if a is None:
                    return 0
                a = str(a).strip().lower()
                if a in ("long", "buy", "1", "+1"):
                    return 1
                if a in ("short", "sell", "-1"):
                    return -1
                return 0

            df["signal"] = df["action"].map(_map)

        if "signal" not in df.columns:
            raise ValueError("signals bundle.df missing 'signal' (or 'position'/'action')")

        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int).clip(-1, 1)
        df = df[df["signal"] != 0].copy()
        return df[["ts", "signal"]].reset_index(drop=True)

    raise ValueError("Unsupported signals bundle shape from load_signals()")


# -----------------------
# Regime CSV -> bricks
# -----------------------
def load_external_regime_to_bricks(
    bars_ts: pd.Series,
    regime_csv: str,
    ts_col: str = "ts",
    value_col: str = "gate_on",
    default_off: bool = True,
) -> pd.Series:
    b = pd.DataFrame({"ts": pd.to_datetime(bars_ts, utc=True, errors="coerce")}).dropna().sort_values("ts")

    r = pd.read_csv(regime_csv)
    if ts_col not in r.columns:
        raise ValueError(f"regime csv missing ts_col='{ts_col}' (cols={list(r.columns)})")
    if value_col not in r.columns:
        raise ValueError(f"regime csv missing value_col='{value_col}' (cols={list(r.columns)})")

    r = r[[ts_col, value_col]].copy()
    r[ts_col] = pd.to_datetime(r[ts_col], utc=True, errors="coerce")
    r = r.dropna(subset=[ts_col]).sort_values(ts_col)
    r = r.drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)

    v = pd.to_numeric(r[value_col], errors="coerce")
    if v.isna().all():
        vv = r[value_col].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
        v = vv
    r["gate"] = v.fillna(0).astype(int).clip(0, 1)

    merged = pd.merge_asof(
        b,
        r[[ts_col, "gate"]].rename(columns={ts_col: "ts"}),
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )

    if default_off:
        merged["gate"] = merged["gate"].fillna(0).astype(int)
    else:
        merged["gate"] = merged["gate"].fillna(1).astype(int)

    out = pd.Series(merged["gate"].astype(bool).values, index=pd.DatetimeIndex(merged["ts"], tz="UTC"), dtype="bool")
    out = out[~out.index.duplicated(keep="last")]
    return out


# -----------------------
# Mapping ts -> fills price
# -----------------------
def map_ts_to_fills_asof(ts_df: pd.DataFrame, fills: pd.DataFrame, fill_col: str) -> Tuple[pd.DataFrame, float]:
    x = ts_df.copy()
    x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
    x = x.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    f = fills.copy()
    f["ts"] = pd.to_datetime(f["ts"], utc=True, errors="coerce")
    f = f.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    merged = pd.merge_asof(
        x,
        f.rename(columns={fill_col: "price_real"}),
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    miss_rate = float(merged["price_real"].isna().mean()) if len(merged) else 0.0
    return merged, miss_rate


# -----------------------
# IMBA opposite-exit engine
# -----------------------
@dataclass
class OppParams:
    fee_bps: float = 10.0
    gate_mode: str = "entries_only"  # entries_only | analysis_only | off


def build_bar_signal_series(bars: pd.DataFrame, signals_df: pd.DataFrame) -> pd.Series:
    bars_ts = pd.to_datetime(bars["ts"], utc=True, errors="coerce")
    bars_ts = pd.DatetimeIndex(bars_ts, tz="UTC")
    sig = pd.Series(0, index=bars_ts, dtype="int64")

    sdf = signals_df.copy()
    sdf["ts"] = pd.to_datetime(sdf["ts"], utc=True, errors="coerce")
    sdf = sdf.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")
    sdf["signal"] = pd.to_numeric(sdf["signal"], errors="coerce").fillna(0).astype(int).clip(-1, 1)

    idx = pd.DatetimeIndex(sdf["ts"].values, tz="UTC")
    hit = idx.intersection(sig.index)

    hit_rate = (len(hit) / max(1, len(idx))) * 100.0
    print(f"INFO signals: n={len(idx)} exact_hits={len(hit)} hit_rate={hit_rate:.2f}%")

    if len(hit) > 0:
        m = sdf.set_index("ts")
        m.index = pd.DatetimeIndex(m.index, tz="UTC")
        sig.loc[hit] = m.loc[hit, "signal"].astype(int).values
        return sig

    left = pd.DataFrame({"ts": bars_ts})
    right = sdf[["ts", "signal"]].copy()
    right["ts"] = pd.to_datetime(right["ts"], utc=True, errors="coerce")
    right = right.dropna(subset=["ts"]).sort_values("ts")
    right["ts"] = pd.DatetimeIndex(right["ts"], tz="UTC")

    tmp = pd.merge_asof(
        left,
        right,
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )

    s2 = tmp["signal"].fillna(0).astype(int).clip(-1, 1).values
    impulses = np.zeros_like(s2)
    last = 0
    for i, v in enumerate(s2):
        if v != 0 and v != last:
            impulses[i] = v
            last = v
    return pd.Series(impulses, index=bars_ts, dtype="int64")


def run_imba_opposite_exit(
    bars: pd.DataFrame,
    signals_df: pd.DataFrame,
    params: OppParams,
    regime_on: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    b = bars.copy().reset_index(drop=True)
    b["ts"] = pd.to_datetime(b["ts"], utc=True, errors="coerce")
    b = b.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    sig_imp = build_bar_signal_series(b, signals_df)
    sig_imp = sig_imp.reindex(pd.DatetimeIndex(b["ts"], tz="UTC"), fill_value=0).astype(int)

    if regime_on is not None:
        gate = regime_on.reindex(sig_imp.index, method="ffill").fillna(False).astype(bool)
    else:
        gate = pd.Series(True, index=sig_imp.index, dtype="bool")

    pos = 0
    last_sig = 0
    events = []
    seq = 0

    def _emit(ts, event, side, price, note, pnl_pct=np.nan):
        nonlocal seq
        events.append(
            {
                "ts": pd.Timestamp(ts),
                "event": str(event),
                "side": int(side),
                "price": float(price) if price is not None else np.nan,
                "pnl_pct": float(pnl_pct) if pnl_pct is not None else np.nan,
                "note": str(note),
                "seq": int(seq),
            }
        )
        seq += 1

    entry_px = None
    entry_ts = None

    for i in range(len(b)):
        ts = pd.Timestamp(b.loc[i, "ts"])
        price = float(b.loc[i, "close"])
        impulse = int(sig_imp.iat[i])
        g = bool(gate.iat[i])

        if impulse == 0:
            continue

        s = int(np.sign(impulse))
        if s == 0:
            continue

        if last_sig == 0:
            allow = True
            if params.gate_mode == "entries_only":
                allow = g
            elif params.gate_mode in ("off", "analysis_only"):
                allow = True

            if allow:
                pos = s
                entry_ts = ts
                entry_px = price
                _emit(ts, "entry", side=s, price=price, note=f"ENTER on first signal {s} gate={int(g)}")
            last_sig = s
            continue

        if s == last_sig:
            continue

        if pos != 0 and entry_px is not None:
            _emit(ts, "signal_flip_exit", side=pos, price=price, note=f"EXIT on opposite signal {s}")
            pos = 0
            entry_px = None
            entry_ts = None

        allow = True
        if params.gate_mode == "entries_only":
            allow = g
        elif params.gate_mode in ("off", "analysis_only"):
            allow = True

        if allow:
            pos = s
            entry_ts = ts
            entry_px = price
            _emit(ts, "entry", side=s, price=price, note=f"ENTER on opposite signal {s} gate={int(g)}")

        last_sig = s

    pos_series = pd.Series(0, index=pd.DatetimeIndex(b["ts"], tz="UTC"), dtype="int64")
    ev_df = pd.DataFrame(events)
    if len(ev_df):
        ev_df["ts"] = pd.to_datetime(ev_df["ts"], utc=True)
        ev_df = ev_df.sort_values(["ts", "seq"], kind="mergesort").reset_index(drop=True)
        cur = 0
        j = 0
        for t in pos_series.index:
            while j < len(ev_df) and ev_df.loc[j, "ts"] <= t:
                if ev_df.loc[j, "event"] == "entry":
                    cur = int(ev_df.loc[j, "side"])
                elif ev_df.loc[j, "event"] == "signal_flip_exit":
                    cur = 0
                j += 1
            pos_series.loc[t] = cur

    return pos_series, ev_df


# -----------------------
# Trades + equity
# -----------------------
def _to_float_scalar(x) -> float:
    """Robust scalar extraction for duplicated columns / pandas oddities."""
    if isinstance(x, pd.Series):
        x = x.iloc[-1]
    if isinstance(x, (list, tuple, np.ndarray)):
        x = x[-1]
    try:
        return float(x)
    except Exception:
        return float("nan")


def pair_trades_from_events(events: pd.DataFrame, price_col: str) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=["entry_ts", "exit_ts", "side", "entry_px", "exit_px", "exit_event"])

    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, errors="coerce")
    ev = ev.dropna(subset=["ts"])
    ev = ev.sort_values(["ts", "seq"], kind="mergesort").reset_index(drop=True)

    open_entry = None
    out = []
    for _, r in ev.iterrows():
        e = str(r.get("event", ""))
        if e == "entry":
            open_entry = r
            continue
        if e == "signal_flip_exit" and open_entry is not None:
            out.append(
                {
                    "entry_ts": pd.Timestamp(open_entry["ts"]),
                    "exit_ts": pd.Timestamp(r["ts"]),
                    "side": int(_to_float_scalar(open_entry.get("side", 0))),
                    "entry_px": _to_float_scalar(open_entry.get(price_col, np.nan)),
                    "exit_px": _to_float_scalar(r.get(price_col, np.nan)),
                    "exit_event": e,
                }
            )
            open_entry = None

    return pd.DataFrame(out)


def compute_trade_pnl(trades: pd.DataFrame, fee_bps: float) -> pd.DataFrame:
    t = trades.copy()
    for c in ["entry_px", "exit_px", "side"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["entry_px", "exit_px", "side"]).reset_index(drop=True)

    fee_rt = 2.0 * float(fee_bps) / 10000.0
    denom = t["entry_px"].abs().replace(0.0, np.nan)
    gross = (t["side"].astype(float) * (t["exit_px"].astype(float) - t["entry_px"].astype(float))) / denom
    t["gross"] = gross.astype(float)
    t["pnl_pct"] = (t["gross"] - fee_rt).astype(float)
    return t


def equity_from_trades(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame({"ts": [], "equity": []})

    t = trades.dropna(subset=["exit_ts", "pnl_pct"]).copy()
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True)
    t["pnl_pct"] = pd.to_numeric(t["pnl_pct"], errors="coerce")
    t = t.dropna(subset=["pnl_pct"]).sort_values("exit_ts").reset_index(drop=True)

    eq = float(initial_capital) * np.cumprod(1.0 + t["pnl_pct"].astype(float).values)
    return pd.DataFrame({"ts": t["exit_ts"].values, "equity": eq})


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak - 1.0) * 100.0
    return float(dd.min())


# -----------------------
# Gate overlap diagnostics
# -----------------------
def position_active_windows(pos: pd.Series) -> pd.Series:
    return (pos.astype(int) != 0).astype(bool)


def overlap_stats(active: pd.Series, gate_on: pd.Series) -> dict:
    a = active.astype(bool)
    g = gate_on.reindex(a.index, method="ffill").fillna(False).astype(bool)
    overlap = float((a & g).mean()) if len(a) else 0.0
    overlap_inv = float((a & (~g)).mean()) if len(a) else 0.0
    return {
        "active_rate": float(a.mean()) if len(a) else 0.0,
        "gate_on_rate": float(g.mean()) if len(g) else 0.0,
        "overlap": overlap,
        "overlap_inv": overlap_inv,
    }


def pnl_split_by_gate(trades: pd.DataFrame, gate_on_bricks: pd.Series) -> dict:
    if trades is None or len(trades) == 0:
        return {"n_on": 0, "n_off": 0, "mean_on": None, "mean_off": None, "sum_on": None, "sum_off": None}

    g = gate_on_bricks.copy()
    g = g[~g.index.duplicated(keep="last")].sort_index()

    t = trades.copy()
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True)
    t = t.sort_values("entry_ts").reset_index(drop=True)

    tmp = pd.merge_asof(
        t[["entry_ts", "pnl_pct"]].rename(columns={"entry_ts": "ts"}),
        g.rename("gate_on").reset_index().rename(columns={"index": "ts"}),
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    tmp["gate_on"] = tmp["gate_on"].fillna(False).astype(bool)

    on = tmp[tmp["gate_on"]]
    off = tmp[~tmp["gate_on"]]
    return {
        "n_on": int(len(on)),
        "n_off": int(len(off)),
        "mean_on": float(on["pnl_pct"].mean()) if len(on) else None,
        "mean_off": float(off["pnl_pct"].mean()) if len(off) else None,
        "sum_on": float(on["pnl_pct"].sum()) if len(on) else None,
        "sum_off": float(off["pnl_pct"].sum()) if len(off) else None,
    }


# -----------------------
# main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renko-parquet", required=True)
    ap.add_argument("--signals-jsonl", required=True)
    ap.add_argument("--fee-bps", type=float, default=10.0)

    ap.add_argument("--fills-parquet", type=str, default=None)
    ap.add_argument("--fill-col", type=str, default="fill_ohlc4")

    ap.add_argument("--regime-csv", type=str, default=None)
    ap.add_argument("--regime-ts-col", type=str, default="ts")
    ap.add_argument("--regime-col", type=str, default="gate_on")
    ap.add_argument("--regime-default-off", action="store_true")

    ap.add_argument("--gate-mode", type=str, default="entries_only", help="entries_only | analysis_only | off")
    ap.add_argument("--run-id", type=str, default=None)

    args = ap.parse_args()

    bars = read_ohlcv_parquet(args.renko_parquet)
    print(f"INFO renko bricks={len(bars)} range={bars['ts'].min()} -> {bars['ts'].max()}")
    print(f"INFO signals jsonl={args.signals_jsonl}")

    bundle = load_signals(path=args.signals_jsonl, kind="jsonl")
    signals_df = signals_bundle_to_df(bundle)

    regime_on = None
    if args.regime_csv:
        regime_on = load_external_regime_to_bricks(
            bars_ts=bars["ts"],
            regime_csv=str(args.regime_csv),
            ts_col=str(args.regime_ts_col),
            value_col=str(args.regime_col),
            default_off=bool(args.regime_default_off),
        )
        print(f"INFO regime csv={args.regime_csv} ON-rate={float(regime_on.mean())*100.0:.2f}%")

    p = OppParams(fee_bps=float(args.fee_bps), gate_mode=str(args.gate_mode).strip().lower())

    pos, events = run_imba_opposite_exit(
        bars=bars[["ts", "open", "high", "low", "close"]].copy(),
        signals_df=signals_df,
        params=p,
        regime_on=(regime_on if args.gate_mode != "off" else None),
    )

    fills_miss_rate = None
    events_used = events.copy()

    if args.fills_parquet:
        fills = read_fills_parquet(str(args.fills_parquet), str(args.fill_col))
        events2, fills_miss_rate = map_ts_to_fills_asof(
            events_used[["ts", "event", "side", "price", "pnl_pct", "note", "seq"]],
            fills,
            str(args.fill_col),
        )
        # IMPORTANT: keep one 'price' column, overwrite with mapped price_real
        events_used = events2.copy()
        events_used["price"] = events_used["price_real"]
        events_used = events_used.drop(columns=["price_real"])
        print(f"INFO fills mapped col={args.fill_col} miss_rate={100.0*float(fills_miss_rate):.4f}%")

    trades = pair_trades_from_events(events_used, price_col="price")
    trades = compute_trade_pnl(trades, fee_bps=float(args.fee_bps))
    equity = equity_from_trades(trades, initial_capital=10_000.0)

    eq0 = 10_000.0
    eq1 = float(equity["equity"].iloc[-1]) if len(equity) else eq0
    ret_pct = (eq1 / eq0 - 1.0) * 100.0
    dd_pct = max_drawdown_pct(equity["equity"]) if len(equity) else 0.0

    stats = {
        "rows": int(len(bars)),
        "start": str(bars["ts"].min()) if len(bars) else None,
        "end": str(bars["ts"].max()) if len(bars) else None,
        "fee_bps": float(args.fee_bps),
        "gate_mode": str(args.gate_mode),
        "regime_csv": (str(args.regime_csv) if args.regime_csv else None),
        "fills_parquet": (str(args.fills_parquet) if args.fills_parquet else None),
        "fill_col": (str(args.fill_col) if args.fills_parquet else None),
        "fills_miss_rate": (float(fills_miss_rate) if fills_miss_rate is not None else None),
        "trades": int(len(trades)),
        "total_return_pct": float(ret_pct),
        "max_drawdown_pct": float(dd_pct),
        "mean_trade_pnl_pct": float(trades["pnl_pct"].mean()) if len(trades) else None,
    }

    if regime_on is not None:
        active = position_active_windows(pos)
        ov = overlap_stats(active=active, gate_on=regime_on)
        split = pnl_split_by_gate(trades, regime_on)
        stats["gate_overlap"] = ov
        stats["pnl_split_by_gate_entry"] = split
        stats["anti_aligned_hint"] = bool(ov["overlap_inv"] > ov["overlap"])

        print(
            f"INFO overlap: active_rate={ov['active_rate']*100:.2f}% gate_on_rate={ov['gate_on_rate']*100:.2f}% "
            f"overlap(active&gate)={ov['overlap']*100:.2f}% overlap(active&~gate)={ov['overlap_inv']*100:.2f}%"
        )
        print(f"INFO pnl split by gate(entry): {split}")

    print(f"INFO stats {stats}")
    if len(events_used):
        last = events_used.sort_values(["ts", "seq"], kind="mergesort").iloc[-1].to_dict()
        print(f"INFO last event: {last}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    pd.DataFrame({"ts": pos.index.values, "pos": pos.astype(int).values}).to_parquet(out_dir / "pos.parquet", index=False)
    events_used.to_parquet(out_dir / "events.parquet", index=False)
    trades.to_parquet(out_dir / "trades_real.parquet", index=False)
    equity.to_parquet(out_dir / "equity_real.parquet", index=False)

    if regime_on is not None:
        pd.DataFrame({"ts": regime_on.index.values, "gate_on": regime_on.astype(int).values}).to_parquet(
            out_dir / "regime.parquet", index=False
        )

    print(f"INFO wrote {out_dir}")


if __name__ == "__main__":
    main()
