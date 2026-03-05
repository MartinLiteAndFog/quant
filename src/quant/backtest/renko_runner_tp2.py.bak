import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("renko_runner_tp2")


# -------------------------
# Params
# -------------------------
@dataclass
class TP2Params:
    tp1_pct: float = 0.015
    tp2_pct: float = 0.030
    tp1_frac: float = 0.5
    min_sl_pct: float = 0.030
    max_sl_pct: float = 0.080
    swing_lookback: int = 180  # effective capped to 50
    flip_on_opposite: bool = True  # wie vorher


# -------------------------
# Time parsing (robust)
# -------------------------
def _parse_ts_any(x) -> pd.Timestamp:
    if x is None:
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        if x.tzinfo is None:
            return x.tz_localize("UTC")
        return x.tz_convert("UTC")
    if isinstance(x, (int, np.integer, float, np.floating)):
        try:
            v = float(x)
        except Exception:
            return pd.NaT
        if not np.isfinite(v):
            return pd.NaT
        av = abs(v)
        if av >= 1e17:
            unit = "ns"
        elif av >= 1e14:
            unit = "us"
        elif av >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        try:
            return pd.to_datetime(int(v), unit=unit, utc=True, errors="coerce")
        except Exception:
            return pd.NaT
    try:
        return pd.to_datetime(x, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def _ensure_utc(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DatetimeIndex):
        s = pd.Series(series)
    else:
        s = series

    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True, errors="coerce")

    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        av = v.abs()
        med = float(av.dropna().median()) if av.notna().any() else 0.0
        if med >= 1e17:
            unit = "ns"
        elif med >= 1e14:
            unit = "us"
        elif med >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(v.astype("Int64"), unit=unit, utc=True, errors="coerce")

    t = s.apply(_parse_ts_any)
    return pd.to_datetime(t, utc=True, errors="coerce")


# -------------------------
# IO
# -------------------------
def _read_renko(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    if "ts" in df.columns:
        df["ts"] = _ensure_utc(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
    else:
        df = df.reset_index().rename(columns={"index": "ts"})
        df["ts"] = _ensure_utc(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)

    cols = {c.lower(): c for c in df.columns}
    for needed in ["open", "high", "low", "close"]:
        if needed not in cols:
            raise ValueError(f"Renko parquet missing column '{needed}'. Got cols={list(df.columns)}")
        df.rename(columns={cols[needed]: needed}, inplace=True)

    if df["ts"].isna().any():
        raise ValueError("Renko parquet contains unparsable ts rows (NaT).")

    return df[["ts", "open", "high", "low", "close"]]


def _load_signals_jsonl(path: str) -> pd.DataFrame:
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(obj, dict):
                bad += 1
                continue

            ts_raw = obj.get("ts", obj.get("t", obj.get("time", None)))
            sig = obj.get("signal", obj.get("pos", obj.get("position", None)))
            if ts_raw is None or sig is None:
                bad += 1
                continue

            try:
                sig_i = int(np.sign(float(sig)))
            except Exception:
                bad += 1
                continue

            if sig_i not in (-1, 0, 1):
                sig_i = int(np.sign(sig_i))

            rows.append((ts_raw, sig_i))

    if not rows:
        raise ValueError(f"No signals loaded from {path}")

    df = pd.DataFrame(rows, columns=["ts", "signal"])
    df["ts"] = _ensure_utc(df["ts"])
    before = len(df)
    df = df.dropna(subset=["ts"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        log.warning(f"signals: dropped {dropped} rows with unparsable ts (NaT) from {path}")
    if bad > 0:
        log.warning(f"signals: skipped {bad} invalid rows from {path}")

    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return df


def _load_gate_external(gate_csv: str, ts_col: str, gate_col: str, default_off: bool) -> pd.DataFrame:
    g = pd.read_csv(gate_csv)
    if ts_col not in g.columns or gate_col not in g.columns:
        raise ValueError(f"Gate CSV missing cols: ts_col={ts_col}, gate_col={gate_col}. Got {list(g.columns)}")
    g[ts_col] = _ensure_utc(g[ts_col])
    g = g.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    gate = pd.to_numeric(g[gate_col], errors="coerce").fillna(0.0)
    gate = (gate > 0.0).astype(int)

    out = pd.DataFrame({"ts": g[ts_col], "gate": gate}).drop_duplicates("ts", keep="last").reset_index(drop=True)
    out.attrs["default_off"] = bool(default_off)
    return out


def _align_gate_to_bricks(bricks: pd.DataFrame, gate_df: pd.DataFrame, default_off: bool) -> pd.Series:
    b = bricks[["ts"]].copy().sort_values("ts")
    g = gate_df[["ts", "gate"]].copy().sort_values("ts")
    merged = pd.merge_asof(b, g, on="ts", direction="backward")
    if default_off:
        return merged["gate"].fillna(0).astype(int)
    return merged["gate"].fillna(1).astype(int)


# -------------------------
# Signal events to bricks (ENTRY ONLY ON EVENT)
# -------------------------
def _signals_to_brick_events(bricks_ts: pd.Series, sig_df: pd.DataFrame) -> pd.Series:
    bt = pd.to_datetime(bricks_ts, utc=True).to_numpy(dtype="datetime64[ns]")
    st = pd.to_datetime(sig_df["ts"], utc=True).to_numpy(dtype="datetime64[ns]")
    sv = sig_df["signal"].to_numpy(dtype=int)

    idx = np.searchsorted(bt, st, side="left")
    m = idx < len(bt)
    idx = idx[m]
    sv = sv[m]

    out = np.zeros(len(bt), dtype=int)
    if len(idx) == 0:
        return pd.Series(out)

    tmp = pd.DataFrame({"i": idx, "s": sv}).groupby("i", as_index=False).last()
    out[tmp["i"].to_numpy()] = tmp["s"].to_numpy(dtype=int)
    return pd.Series(out)


# -------------------------
# Math helpers
# -------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _compute_sl_price(
    side: int,
    entry_px: float,
    swing_low_prev: float,
    swing_high_prev: float,
    min_sl_pct: float,
    max_sl_pct: float,
) -> Tuple[float, float]:
    if side > 0:
        raw = (entry_px - swing_low_prev) / entry_px if entry_px > 0 else max_sl_pct
        sl_pct = _clamp(raw, min_sl_pct, max_sl_pct)
        sl_px = entry_px * (1.0 - sl_pct)
    else:
        raw = (swing_high_prev - entry_px) / entry_px if entry_px > 0 else max_sl_pct
        sl_pct = _clamp(raw, min_sl_pct, max_sl_pct)
        sl_px = entry_px * (1.0 + sl_pct)
    return sl_px, sl_pct


# -------------------------
# Engine: Legs + Trades (Trade = entry -> final flat)
# -------------------------
def run_tp2_engine(
    bricks: pd.DataFrame,
    sig_event: pd.Series,
    gate_on: pd.Series,
    params: TP2Params,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(bricks)
    eff_lb = int(min(max(params.swing_lookback, 1), 50))

    swing_low = bricks["low"].rolling(eff_lb, min_periods=1).min().shift(1).fillna(bricks["low"])
    swing_high = bricks["high"].rolling(eff_lb, min_periods=1).max().shift(1).fillna(bricks["high"])

    events: List[Dict] = []
    legs: List[Dict] = []

    trade_id = 0
    pos_side = 0
    entry_ts: Optional[pd.Timestamp] = None
    entry_px: float = 0.0
    size_rem: float = 0.0
    tp1_done: bool = False
    be_active: bool = False

    seq = 0
    entries = 0
    flips = 0
    sl_exits = 0
    be_exits = 0
    tp1_exits = 0
    tp2_exits = 0
    sig_exits = 0
    regime_exits = 0

    def emit(ts, event, side, price, note, size=None):
        nonlocal seq
        events.append(
            dict(
                ts=ts,
                event=event,
                side=int(side),
                price=float(price),
                size=(float(size) if size is not None else np.nan),
                note=note,
                seq=seq,
                trade_id=(trade_id if pos_side != 0 else np.nan),
            )
        )
        seq += 1

    def open_trade(ts, side, px, note):
        nonlocal trade_id, pos_side, entry_ts, entry_px, size_rem, tp1_done, be_active, entries
        trade_id += 1
        pos_side = int(side)
        entry_ts = ts
        entry_px = float(px)
        size_rem = 1.0
        tp1_done = False
        be_active = False
        entries += 1
        emit(ts, "entry", pos_side, px, note, size=1.0)

    def close_leg(exit_ts, exit_px, exit_event, frac, note=""):
        nonlocal pos_side, entry_ts, entry_px, size_rem, tp1_done, be_active
        frac = float(frac)
        if entry_ts is None or pos_side == 0 or frac <= 0:
            return
        frac = min(frac, size_rem)
        pnl_unw = pos_side * (float(exit_px) / float(entry_px) - 1.0)
        pnl_w = frac * pnl_unw
        legs.append(
            dict(
                trade_id=int(trade_id),
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                side=int(pos_side),
                entry_px=float(entry_px),
                exit_px=float(exit_px),
                exit_event=str(exit_event),
                size=float(frac),
                pnl_pct_unweighted=float(pnl_unw),
                pnl_pct=float(pnl_w),
                note=note,
            )
        )
        size_rem -= frac
        if size_rem <= 1e-9:
            pos_side = 0
            entry_ts = None
            entry_px = 0.0
            size_rem = 0.0
            tp1_done = False
            be_active = False

    def close_trade_all(exit_ts, exit_px, exit_event, note):
        if pos_side == 0:
            return
        close_leg(exit_ts, exit_px, exit_event, frac=size_rem, note=note)
        emit(exit_ts, exit_event, 0, exit_px, note)

    def be_price() -> float:
        return float(entry_px)  # BE on 0

    for i in range(n):
        ts = bricks.at[i, "ts"]
        h = float(bricks.at[i, "high"])
        l = float(bricks.at[i, "low"])
        c = float(bricks.at[i, "close"])

        g = int(gate_on.iat[i])
        ev = int(sig_event.iat[i])

        if g == 0:
            if pos_side != 0:
                regime_exits += 1
                close_trade_all(ts, c, "regime_exit", "Regime off -> flat")
            continue

        if pos_side == 0:
            if ev != 0:
                open_trade(ts, ev, c, "Enter on IMBA SIGNAL EVENT")
            continue

        if be_active:
            bep = be_price()
            if (pos_side > 0 and l <= bep) or (pos_side < 0 and h >= bep):
                be_exits += 1
                close_trade_all(ts, bep, "be_exit", "BE hit (armed after TP1) -> flat")
                continue

        sl_px, sl_pct = _compute_sl_price(
            pos_side,
            entry_px,
            float(swing_low.iat[i]),
            float(swing_high.iat[i]),
            params.min_sl_pct,
            params.max_sl_pct,
        )

        if (pos_side > 0 and l <= sl_px) or (pos_side < 0 and h >= sl_px):
            sl_exits += 1
            close_trade_all(ts, sl_px, "sl_exit", f"SL hit -> flat (sl_pct={sl_pct:.5f})")
            continue

        tp1_px = entry_px * (1.0 + params.tp1_pct) if pos_side > 0 else entry_px * (1.0 - params.tp1_pct)
        tp2_px = entry_px * (1.0 + params.tp2_pct) if pos_side > 0 else entry_px * (1.0 - params.tp2_pct)
        tp1_hit = (pos_side > 0 and h >= tp1_px) or (pos_side < 0 and l <= tp1_px)
        tp2_hit = (pos_side > 0 and h >= tp2_px) or (pos_side < 0 and l <= tp2_px)

        if tp2_hit and size_rem > 0:
            tp2_exits += 1
            close_trade_all(ts, tp2_px, "tp2_exit", "TP2 hit -> flat")
            continue

        if (not tp1_done) and tp1_hit and size_rem > (1.0 - params.tp1_frac + 1e-9):
            frac = _clamp(params.tp1_frac, 0.0, 1.0)
            tp1_done = True
            tp1_exits += 1
            close_leg(ts, tp1_px, "tp1_exit", frac=frac, note=f"TP1 hit -> scale out frac={frac:.2f}")
            emit(ts, "tp1_exit", pos_side, tp1_px, f"TP1 hit -> scale out frac={frac:.2f}", size=frac)

            be_active = True
            emit(ts, "be_armed", pos_side, be_price(), "BE armed at entry after TP1")

        if ev != 0 and ev == -pos_side:
            sig_exits += 1
            close_trade_all(ts, c, "signal_exit", "Opposite IMBA signal -> close trade")
            if params.flip_on_opposite:
                flips += 1
                open_trade(ts, ev, c, "Flip: open opposite on same bar close")
            continue

    if pos_side != 0 and entry_ts is not None:
        close_trade_all(bricks.at[n - 1, "ts"], float(bricks.at[n - 1, "close"]), "eod_exit", "End of data -> flat")

    events_df = pd.DataFrame(events)
    legs_df = pd.DataFrame(legs)

    events_df.attrs["counts"] = dict(
        trades_entries=entries,
        trades_flips=flips,
        sl_exits=sl_exits,
        be_exits=be_exits,
        tp1_exits=tp1_exits,
        tp2_exits=tp2_exits,
        signal_exits=sig_exits,
        regime_exits=regime_exits,
        swing_lookback_effective=eff_lb,
    )
    return events_df, legs_df


def legs_to_trades(legs_df: pd.DataFrame) -> pd.DataFrame:
    if len(legs_df) == 0:
        return pd.DataFrame()

    df = legs_df.copy()
    df["entry_ts"] = _ensure_utc(df["entry_ts"])
    df["exit_ts"] = _ensure_utc(df["exit_ts"])

    grp = df.groupby("trade_id", sort=False)
    trades = grp.agg(
        entry_ts=("entry_ts", "min"),
        exit_ts=("exit_ts", "max"),
        side=("side", "first"),
        entry_px=("entry_px", "first"),
        exit_px=("exit_px", "last"),
        exit_event=("exit_event", "last"),
        legs=("exit_ts", "size"),
        size_sum=("size", "sum"),
        pnl_pct=("pnl_pct", "sum"),
    ).reset_index()

    return trades.sort_values("exit_ts").reset_index(drop=True)


def map_legs_to_fills(legs_df: pd.DataFrame, fills_parquet: str, fill_col: str) -> pd.DataFrame:
    f = pd.read_parquet(fills_parquet, columns=["ts", fill_col])
    f["ts"] = _ensure_utc(f["ts"])
    f = f.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    f.rename(columns={fill_col: "fill_px"}, inplace=True)

    t = legs_df.copy().reset_index(drop=True)
    t["entry_ts"] = _ensure_utc(t["entry_ts"])
    t["exit_ts"] = _ensure_utc(t["exit_ts"])
    t = t.dropna(subset=["entry_ts", "exit_ts"]).reset_index(drop=True)
    t["_i"] = np.arange(len(t), dtype=int)

    te = t[["_i", "entry_ts"]].sort_values("entry_ts").reset_index(drop=True)
    me = pd.merge_asof(te, f, left_on="entry_ts", right_on="ts", direction="backward")[["_i", "fill_px"]]
    me = me.rename(columns={"fill_px": "entry_px_real"})

    tx = t[["_i", "exit_ts"]].sort_values("exit_ts").reset_index(drop=True)
    mx = pd.merge_asof(tx, f, left_on="exit_ts", right_on="ts", direction="backward")[["_i", "fill_px"]]
    mx = mx.rename(columns={"fill_px": "exit_px_real"})

    t = t.merge(me, on="_i", how="left").merge(mx, on="_i", how="left")
    miss = float(np.mean(t["entry_px_real"].isna() | t["exit_px_real"].isna()))
    t.attrs["fills_miss_rate"] = miss
    if miss > 0:
        t = t.dropna(subset=["entry_px_real", "exit_px_real"]).reset_index(drop=True)

    side = pd.to_numeric(t["side"], errors="coerce").astype(int).to_numpy()
    size = pd.to_numeric(t["size"], errors="coerce").astype(float).to_numpy()
    ep = pd.to_numeric(t["entry_px_real"], errors="coerce").astype(float).to_numpy()
    xp = pd.to_numeric(t["exit_px_real"], errors="coerce").astype(float).to_numpy()

    pnl_unw = side * (xp / ep - 1.0)
    pnl_w = size * pnl_unw

    t["pnl_pct_real_unweighted"] = pnl_unw
    t["pnl_pct_real"] = pnl_w
    return t.drop(columns=["_i"], errors="ignore")


def equity_from_pnls(pnls: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + pnls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--box", type=float, default=0.1)
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--signals-jsonl", required=True)
    ap.add_argument("--run-id", default=None)

    ap.add_argument("--tp1-pct", type=float, default=0.015)
    ap.add_argument("--tp2-pct", type=float, default=0.030)
    ap.add_argument("--tp1-frac", type=float, default=0.5)

    ap.add_argument("--min-sl-pct", type=float, default=0.030)
    ap.add_argument("--max-sl-pct", type=float, default=0.080)
    ap.add_argument("--swing-lookback", type=int, default=180)

    ap.add_argument("--no-flip-on-opposite", action="store_true")

    ap.add_argument("--regime", default="none")
    ap.add_argument("--regime-csv", default=None)
    ap.add_argument("--regime-ts-col", default="ts")
    ap.add_argument("--regime-col", default="gate_on")
    ap.add_argument("--regime-default-off", action="store_true")

    ap.add_argument("--fills-parquet", default=None)
    ap.add_argument("--fill-col", default=None)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_id = args.run_id or "TP2_run"
    out_dir = os.path.join("data", "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)

    bricks = _read_renko(args.parquet)
    log.info(f"renko box={args.box} bricks={len(bricks)}")

    sig_df = _load_signals_jsonl(args.signals_jsonl)
    log.info(f"signals jsonl={args.signals_jsonl} parsed={len(sig_df)} range={sig_df.ts.iloc[0]} -> {sig_df.ts.iloc[-1]}")

    sig_event = _signals_to_brick_events(bricks["ts"], sig_df)

    if args.regime == "external":
        if not args.regime_csv:
            raise ValueError("--regime external requires --regime-csv")
        gate_df = _load_gate_external(args.regime_csv, args.regime_ts_col, args.regime_col, args.regime_default_off)
        gate_on = _align_gate_to_bricks(bricks, gate_df, args.regime_default_off)
        log.info(f"regime=external csv={args.regime_csv} ON-rate={float(gate_on.mean())*100:.2f}%")
    else:
        gate_on = pd.Series(np.ones(len(bricks), dtype=int))

    params = TP2Params(
        tp1_pct=float(args.tp1_pct),
        tp2_pct=float(args.tp2_pct),
        tp1_frac=float(args.tp1_frac),
        min_sl_pct=float(args.min_sl_pct),
        max_sl_pct=float(args.max_sl_pct),
        swing_lookback=int(args.swing_lookback),
        flip_on_opposite=(not args.no_flip_on_opposite),
    )

    events_df, legs_df = run_tp2_engine(bricks, sig_event, gate_on, params)
    counts = events_df.attrs.get("counts", {})

    if len(legs_df) == 0:
        raise RuntimeError("No legs produced. Check: gate_on, signal event mapping, etc.")

    trades_df = legs_to_trades(legs_df)
    pnl_tr = trades_df["pnl_pct"].astype(float).to_numpy()
    eq = equity_from_pnls(pnl_tr)
    total_return = float(eq[-1] - 1.0)
    mdd = _max_drawdown(eq)

    fee_rt = float(args.fee_bps) / 10000.0
    eq_net = equity_from_pnls(pnl_tr - fee_rt)
    total_return_net = float(eq_net[-1] - 1.0)
    mdd_net = _max_drawdown(eq_net)

    fills_miss = 0.0
    legs_real_df = None
    trades_real_df = None
    eq_real = None
    eq_real_net = None

    if args.fills_parquet and args.fill_col:
        legs_real_df = map_legs_to_fills(legs_df, args.fills_parquet, args.fill_col)
        fills_miss = float(legs_real_df.attrs.get("fills_miss_rate", 0.0))
        log.info(f"fills mapped: col={args.fill_col} miss_rate={fills_miss*100:.4f}%")

        tmp = legs_real_df.copy()
        tmp["pnl_pct"] = tmp["pnl_pct_real"]
        trades_real_df = legs_to_trades(tmp)

        pnl_real = trades_real_df["pnl_pct"].astype(float).to_numpy()
        eq_real = equity_from_pnls(pnl_real)
        eq_real_net = equity_from_pnls(pnl_real - fee_rt)

    stats = {
        "rows": int(len(bricks)),
        "start": str(bricks["ts"].iloc[0]),
        "end": str(bricks["ts"].iloc[-1]),
        "legs": int(len(legs_df)),
        "trades": int(len(trades_df)),
        "total_return_pct": total_return * 100.0,
        "max_drawdown_pct": mdd * 100.0,
        "total_return_pct_net": total_return_net * 100.0,
        "max_drawdown_pct_net": mdd_net * 100.0,
        "fee_bps_roundtrip": float(args.fee_bps),
        "tp1_pct": float(params.tp1_pct),
        "tp2_pct": float(params.tp2_pct),
        "tp1_frac": float(params.tp1_frac),
        "min_sl_pct": float(params.min_sl_pct),
        "max_sl_pct": float(params.max_sl_pct),
        "swing_lookback_effective": int(counts.get("swing_lookback_effective", 50)),
        "flip_on_opposite": bool(params.flip_on_opposite),
        "entries": int(counts.get("trades_entries", 0)),
        "flips": int(counts.get("trades_flips", 0)),
        "sl_exits": int(counts.get("sl_exits", 0)),
        "be_exits": int(counts.get("be_exits", 0)),
        "tp1_exits": int(counts.get("tp1_exits", 0)),
        "tp2_exits": int(counts.get("tp2_exits", 0)),
        "signal_exits": int(counts.get("signal_exits", 0)),
        "regime_exits": int(counts.get("regime_exits", 0)),
        "regime": str(args.regime),
        "regime_csv": args.regime_csv,
        "regime_col": args.regime_col,
        "signals_jsonl": args.signals_jsonl,
        "parquet": args.parquet,
        "box": float(args.box),
        "fills_parquet": args.fills_parquet,
        "fill_col": args.fill_col,
        "fills_miss_rate": float(fills_miss),
    }

    if trades_real_df is not None:
        stats.update(
            {
                "total_return_pct_real": float(eq_real[-1] - 1.0) * 100.0,
                "max_drawdown_pct_real": _max_drawdown(eq_real) * 100.0,
                "total_return_pct_real_net": float(eq_real_net[-1] - 1.0) * 100.0,
                "max_drawdown_pct_real_net": _max_drawdown(eq_real_net) * 100.0,
            }
        )

    log.info(f"stats {stats}")

    events_df.to_parquet(os.path.join(out_dir, "events.parquet"), index=False)
    legs_df.to_parquet(os.path.join(out_dir, "legs.parquet"), index=False)
    trades_df.to_parquet(os.path.join(out_dir, "trades.parquet"), index=False)

    pd.DataFrame({"ts": trades_df["exit_ts"], "equity": eq}).to_parquet(os.path.join(out_dir, "equity.parquet"), index=False)
    pd.DataFrame({"ts": trades_df["exit_ts"], "equity": eq_net}).to_parquet(os.path.join(out_dir, "equity_net.parquet"), index=False)

    pd.DataFrame({"ts": bricks["ts"], "gate_on": gate_on.astype(int)}).to_parquet(
        os.path.join(out_dir, "regime.parquet"), index=False
    )

    if legs_real_df is not None:
        legs_real_df.to_parquet(os.path.join(out_dir, "legs_real.parquet"), index=False)
    if trades_real_df is not None:
        trades_real_df.to_parquet(os.path.join(out_dir, "trades_real.parquet"), index=False)
        pd.DataFrame({"ts": trades_real_df["exit_ts"], "equity": eq_real}).to_parquet(os.path.join(out_dir, "equity_real.parquet"), index=False)
        pd.DataFrame({"ts": trades_real_df["exit_ts"], "equity": eq_real_net}).to_parquet(os.path.join(out_dir, "equity_real_net.parquet"), index=False)

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    log.info(f"wrote {out_dir}")


if __name__ == "__main__":
    main()