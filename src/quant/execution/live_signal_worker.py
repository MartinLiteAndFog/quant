from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.execution.execution_state import write_execution_state
from quant.execution.kucoin_futures import KucoinFuturesBroker, _symbol_to_contract
from quant.execution.strategy_router import strategy_for_gate, trend_signals_from_imba
from quant.features.renko import renko_from_close
from quant.regime import RegimeStateRecord, RegimeStore
from quant.strategies.imba import ImbaParams, compute_imba_signals
from quant.utils.log import get_logger

log = get_logger("quant.live_signal_worker")


@dataclass
class WorkerState:
    last_signal_ts: Optional[str] = None
    last_countertrend_ts: Optional[str] = None
    last_trendfollower_ts: Optional[str] = None
    last_poll_ts: Optional[str] = None
    n_emitted: int = 0


def _normalize_symbol(sym: str) -> str:
    s = sym.strip().upper().replace("/", "-").replace(":", "-").replace(" ", "")
    return s or "UNKNOWN"


def _today_utc() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y%m%d")


def _now_utc_iso() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_state(path: Path) -> WorkerState:
    if not path.exists():
        return WorkerState()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return WorkerState(
            last_signal_ts=obj.get("last_signal_ts"),
            last_countertrend_ts=obj.get("last_countertrend_ts"),
            last_trendfollower_ts=obj.get("last_trendfollower_ts"),
            last_poll_ts=obj.get("last_poll_ts"),
            n_emitted=int(obj.get("n_emitted", 0)),
        )
    except Exception:
        return WorkerState()


def _write_state(path: Path, st: WorkerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(st), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _parse_kucoin_1m_rows(rows: List[List[Any]]) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 5:
            continue
        try:
            ts_i = int(float(r[0]))
            ts = pd.to_datetime(ts_i, unit="ms" if ts_i > 10**12 else "s", utc=True)
            out.append(
                {
                    "ts": ts,
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                }
            )
        except Exception:
            continue
    if not out:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    df = pd.DataFrame(out)
    return df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)


def _fetch_recent_1m_ohlcv(broker: KucoinFuturesBroker, symbol: str, limit: int) -> pd.DataFrame:
    contract = _symbol_to_contract(symbol)
    lim = int(max(1, limit))
    now = pd.Timestamp.now("UTC")
    start = now - pd.Timedelta(minutes=lim + 30)
    step = pd.Timedelta(hours=2)

    chunks: List[pd.DataFrame] = []
    cur = start
    while cur < now:
        nxt = min(cur + step, now)
        from_ms = int(cur.timestamp() * 1000)
        to_ms = int(nxt.timestamp() * 1000)
        data = broker._req(
            "GET",
            f"/api/v1/kline/query?symbol={contract}&granularity=1&from={from_ms}&to={to_ms}",
        )
        rows = data if isinstance(data, list) else (data.get("data", []) if isinstance(data, dict) else [])
        df_page = _parse_kucoin_1m_rows(rows)
        if not df_page.empty:
            df_page = df_page[(df_page["ts"] >= cur) & (df_page["ts"] < nxt)]
            if not df_page.empty:
                chunks.append(df_page)
        cur = nxt

    if not chunks:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return df.iloc[-lim:].reset_index(drop=True) if len(df) > lim else df


def _renko_to_ohlc(bricks: pd.DataFrame) -> pd.DataFrame:
    b = bricks.copy()
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    b["open"] = pd.to_numeric(b["open"], errors="coerce")
    b["close"] = pd.to_numeric(b["close"], errors="coerce")
    b = b.dropna(subset=["ts", "open", "close"]).sort_values("ts").reset_index(drop=True)
    if b.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])

    out = pd.DataFrame(
        {
            "ts": b["ts"].values,
            "open": b["open"].values,
            "high": b[["open", "close"]].max(axis=1).values,
            "low": b[["open", "close"]].min(axis=1).values,
            "close": b["close"].values,
        }
    )
    if len(out) > 1:
        dup = out["ts"].duplicated(keep=False)
        if dup.any():
            grp = out["ts"].astype("int64")
            idx_in_grp = out.groupby(grp).cumcount()
            out["ts"] = out["ts"] + pd.to_timedelta(idx_in_grp, unit="ns")
    return out


def _append_signal_jsonl(out_path: Path, rec: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def _last_jsonl_record(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def _append_signal_jsonl_dedupe(out_path: Path, rec: Dict[str, Any]) -> bool:
    prev = _last_jsonl_record(out_path)
    if prev:
        same_ts = str(prev.get("ts")) == str(rec.get("ts"))
        same_sig = int(prev.get("signal", 0)) == int(rec.get("signal", 0))
        same_mode = str(prev.get("strategy_mode", "")) == str(rec.get("strategy_mode", ""))
        if same_ts and same_sig and same_mode:
            return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
    return True


def _filter_after(df: pd.DataFrame, ts_iso: Optional[str]) -> pd.DataFrame:
    if df.empty or not ts_iso:
        return df
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"])
    ts = pd.to_datetime(ts_iso, utc=True, errors="coerce")
    if pd.isna(ts):
        return out
    return out[out["ts"] > ts].copy()


def _load_or_seed_gate(regime_store: RegimeStore, symbol: str, default_gate_on: int) -> Dict[str, Any]:
    latest = regime_store.get_latest_state(symbol=symbol)
    if latest:
        return latest
    now = _now_utc_iso()
    gate_on = int(default_gate_on)
    regime_store.upsert_regime_state(
        RegimeStateRecord(
            ts=now,
            symbol=symbol,
            gate_on=gate_on,
            regime_state=strategy_for_gate(gate_on),
            regime_score=1.0 if gate_on else -1.0,
            confidence=0.5,
            reason_code="live_signal_default_gate_seed",
            model_version="live-signal-v1",
            feature_values_json="{}",
        )
    )
    return regime_store.get_latest_state(symbol=symbol) or {"gate_on": gate_on, "regime_state": strategy_for_gate(gate_on)}


def run_once(
    broker: KucoinFuturesBroker,
    *,
    symbol: str,
    renko_box: float,
    lookback: int,
    sl_abs: float,
    candles_limit: int,
    signals_dir: Path,
    regime_store: RegimeStore,
    default_gate_on: int,
    state: WorkerState,
) -> WorkerState:
    bars = _fetch_recent_1m_ohlcv(broker, symbol=symbol, limit=candles_limit)
    if len(bars) < max(lookback, 20):
        log.info("live-signal bars=%s waiting for enough data", len(bars))
        state.last_poll_ts = _now_utc_iso()
        return state

    bricks = renko_from_close(bars[["ts", "close"]], box=float(renko_box))
    if bricks.empty:
        log.info("live-signal no renko bricks (box=%s)", renko_box)
        state.last_poll_ts = _now_utc_iso()
        return state

    renko_ohlc = _renko_to_ohlc(bricks)
    imba_all = compute_imba_signals(
        renko_ohlc,
        ImbaParams(
            lookback=int(lookback),
            fixed_sl_abs=float(sl_abs),
        ),
    ).sort_values("ts").reset_index(drop=True)
    if imba_all.empty:
        state.last_poll_ts = _now_utc_iso()
        return state

    trend_all = trend_signals_from_imba(imba_all).sort_values("ts").reset_index(drop=True)
    gate = _load_or_seed_gate(regime_store=regime_store, symbol=symbol, default_gate_on=default_gate_on)
    gate_on = int(gate.get("gate_on", default_gate_on))
    active_mode = strategy_for_gate(gate_on)
    regime_store.upsert_regime_state(
        RegimeStateRecord(
            ts=_now_utc_iso(),
            symbol=symbol,
            gate_on=gate_on,
            regime_state=active_mode,
            regime_score=1.0 if gate_on else -1.0,
            confidence=float(gate.get("confidence", 0.7 if gate_on else 0.6)),
            reason_code="live_signal_heartbeat",
            model_version="live-signal-v1",
            feature_values_json=json.dumps(
                {
                    "lookback": int(lookback),
                    "bars_available": int(len(renko_ohlc)),
                    "active_mode": active_mode,
                },
                separators=(",", ":"),
            ),
        )
    )

    imba_new = _filter_after(imba_all, state.last_countertrend_ts)
    trend_new = _filter_after(trend_all, state.last_trendfollower_ts)
    active_base = imba_all if active_mode == "countertrend" else trend_all
    active_new = _filter_after(active_base, state.last_signal_ts)

    sym_norm = _normalize_symbol(symbol)
    root_path = signals_dir / sym_norm / f"{_today_utc()}.jsonl"
    imba_path = signals_dir / sym_norm / "countertrend" / f"{_today_utc()}.jsonl"
    trend_path = signals_dir / sym_norm / "trendfollower" / f"{_today_utc()}.jsonl"

    for _, r in imba_new.iterrows():
        rec = {
            "server_ts": _now_utc_iso(),
            "ts": pd.Timestamp(r["ts"], tz="UTC").isoformat(),
            "signal": int(r["signal"]),
            "position": int(r.get("position", r["signal"])),
            "source": "imba_live_worker",
            "strategy_mode": "countertrend",
            "sl": float(r["sl"]) if not pd.isna(r.get("sl")) else None,
            "symbol": symbol,
            "gate_on": gate_on,
            "lookback": int(lookback),
        }
        _append_signal_jsonl_dedupe(imba_path, rec)
        state.last_countertrend_ts = rec["ts"]

    for _, r in trend_new.iterrows():
        rec = {
            "server_ts": _now_utc_iso(),
            "ts": pd.Timestamp(r["ts"], tz="UTC").isoformat(),
            "signal": int(r["signal"]),
            "position": int(r.get("position", r["signal"])),
            "source": "imba_live_worker",
            "strategy_mode": "trendfollower",
            "sl": float(r["sl"]) if not pd.isna(r.get("sl")) else None,
            "symbol": symbol,
            "gate_on": gate_on,
            "lookback": int(lookback),
        }
        _append_signal_jsonl_dedupe(trend_path, rec)
        state.last_trendfollower_ts = rec["ts"]

    emitted_active = 0
    for _, r in active_new.iterrows():
        rec = {
            "server_ts": _now_utc_iso(),
            "ts": pd.Timestamp(r["ts"], tz="UTC").isoformat(),
            "signal": int(r["signal"]),
            "position": int(r.get("position", r["signal"])),
            "source": "imba_live_worker",
            "strategy_mode": active_mode,
            "sl": float(r["sl"]) if not pd.isna(r.get("sl")) else None,
            "symbol": symbol,
            "gate_on": gate_on,
            "lookback": int(lookback),
        }
        wrote = _append_signal_jsonl_dedupe(root_path, rec)
        if wrote:
            state.last_signal_ts = rec["ts"]
            state.n_emitted += 1
            emitted_active += 1
            log.info(
                "live-signal emitted symbol=%s strategy=%s gate_on=%s ts=%s signal=%s file=%s",
                symbol,
                active_mode,
                gate_on,
                rec["ts"],
                rec["signal"],
                root_path,
            )

    latest_active = active_base.iloc[-1] if len(active_base) else None
    write_execution_state(
        {
            "symbol": symbol,
            "mode": active_mode,
            "gate_on": gate_on,
            "regime_state": gate.get("regime_state"),
            "lookback": int(lookback),
            "bars_available": int(len(renko_ohlc)),
            "signal": int(latest_active["signal"]) if latest_active is not None else None,
            "sl": float(latest_active["sl"]) if latest_active is not None and not pd.isna(latest_active.get("sl")) else None,
            "ts": pd.Timestamp(latest_active["ts"], tz="UTC").isoformat() if latest_active is not None else _now_utc_iso(),
        }
    )
    latest_calc = active_base.iloc[-1] if len(active_base) else None
    if latest_calc is not None:
        log.info(
            "live-signal status symbol=%s strategy=%s gate_on=%s latest_calc_ts=%s latest_calc_sig=%s emitted_now=%s",
            symbol,
            active_mode,
            gate_on,
            pd.Timestamp(latest_calc["ts"], tz="UTC").isoformat(),
            int(latest_calc["signal"]),
            emitted_active,
        )

    state.last_poll_ts = _now_utc_iso()
    return state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run live strategy signal worker from KuCoin Futures 1m candles")
    p.add_argument("--symbol", default=os.getenv("LIVE_SYMBOL", "SOL-USDT"))
    p.add_argument("--renko-box", type=float, default=float(os.getenv("LIVE_RENKO_BOX", "0.1")))
    p.add_argument("--lookback", type=int, default=int(os.getenv("LIVE_IMBA_LOOKBACK", "250")))
    p.add_argument("--sl-abs", type=float, default=float(os.getenv("LIVE_IMBA_SL_ABS", "1.5")))
    p.add_argument("--candles-limit", type=int, default=int(os.getenv("LIVE_CANDLES_LIMIT", "1500")))
    p.add_argument("--poll-sec", type=float, default=float(os.getenv("LIVE_POLL_SEC", "15")))
    p.add_argument("--signals-dir", default=os.getenv("SIGNALS_DIR", "data/signals"))
    p.add_argument("--state-file", default=os.getenv("LIVE_SIGNAL_STATE", "data/live/live_signal_state.json"))
    p.add_argument("--default-gate-on", type=int, default=int(os.getenv("LIVE_DEFAULT_GATE_ON", "1")))
    p.add_argument("--once", action="store_true", help="Run one cycle and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    broker = KucoinFuturesBroker()
    signals_dir = Path(args.signals_dir)
    state_path = Path(args.state_file)
    regime_store = RegimeStore()
    st = _read_state(state_path)

    while True:
        try:
            st = run_once(
                broker,
                symbol=args.symbol,
                renko_box=float(args.renko_box),
                lookback=int(args.lookback),
                sl_abs=float(args.sl_abs),
                candles_limit=int(args.candles_limit),
                signals_dir=signals_dir,
                regime_store=regime_store,
                default_gate_on=int(args.default_gate_on),
                state=st,
            )
            _write_state(state_path, st)
        except Exception as e:
            log.warning("live-signal worker loop error: %s", e)
            st.last_poll_ts = _now_utc_iso()
            _write_state(state_path, st)

        if args.once:
            break
        time.sleep(max(1.0, float(args.poll_sec)))


if __name__ == "__main__":
    main()
