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
from quant.features.renko import renko_from_close
from quant.strategies.imba import ImbaParams, compute_imba_signals
from quant.utils.log import get_logger

log = get_logger("quant.live_signal_worker")


@dataclass
class WorkerState:
    last_signal_ts: Optional[str] = None
    last_poll_ts: Optional[str] = None
    n_emitted: int = 0


def _normalize_symbol(sym: str) -> str:
    s = sym.strip().replace("/", "-").replace(":", "-").replace(" ", "")
    return s or "UNKNOWN"


def _today_utc() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d")


def _now_utc_iso() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_state(path: Path) -> WorkerState:
    if not path.exists():
        return WorkerState()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return WorkerState(
            last_signal_ts=obj.get("last_signal_ts"),
            last_poll_ts=obj.get("last_poll_ts"),
            n_emitted=int(obj.get("n_emitted", 0)),
        )
    except Exception:
        return WorkerState()


def _write_state(path: Path, st: WorkerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(st), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _parse_kucoin_1m_rows(rows: List[List[Any]]) -> pd.DataFrame:
    """
    Parse KuCoin kline rows into OHLC dataframe.
    Expected row shape: [time, open, high, low, close, ...].
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 5:
            continue
        ts_raw = r[0]
        try:
            ts_i = int(float(ts_raw))
            # KuCoin can return sec or ms; infer from magnitude.
            if ts_i > 10**12:
                ts = pd.to_datetime(ts_i, unit="ms", utc=True)
            else:
                ts = pd.to_datetime(ts_i, unit="s", utc=True)
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
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return df


def _fetch_recent_1m_ohlcv(broker: KucoinFuturesBroker, symbol: str, limit: int) -> pd.DataFrame:
    contract = _symbol_to_contract(symbol)
    data = broker._req("GET", f"/api/v1/kline/query?symbol={contract}&granularity=60&from=0")
    rows = data if isinstance(data, list) else (data.get("data", []) if isinstance(data, dict) else [])
    df = _parse_kucoin_1m_rows(rows)
    if limit > 0 and len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df


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


def run_once(
    broker: KucoinFuturesBroker,
    *,
    symbol: str,
    renko_box: float,
    lookback: int,
    sl_abs: float,
    candles_limit: int,
    signals_dir: Path,
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
    sig = compute_imba_signals(
        renko_ohlc,
        ImbaParams(
            lookback=int(lookback),
            fixed_sl_abs=float(sl_abs),
        ),
    )
    if sig.empty:
        state.last_poll_ts = _now_utc_iso()
        return state

    sig = sig.sort_values("ts").reset_index(drop=True)
    if state.last_signal_ts:
        last_ts = pd.to_datetime(state.last_signal_ts, utc=True, errors="coerce")
        if pd.notna(last_ts):
            sig = sig[sig["ts"] > last_ts].copy()

    if sig.empty:
        state.last_poll_ts = _now_utc_iso()
        return state

    sym_norm = _normalize_symbol(symbol)
    out_path = signals_dir / sym_norm / f"{_today_utc()}.jsonl"

    for _, r in sig.iterrows():
        ts = pd.Timestamp(r["ts"], tz="UTC")
        rec = {
            "server_ts": _now_utc_iso(),
            "ts": ts.isoformat(),
            "signal": int(r["signal"]),
            "position": int(r.get("position", r["signal"])),
            "source": "imba_live_worker",
            "sl": float(r["sl"]) if not pd.isna(r.get("sl")) else None,
            "symbol": symbol,
        }
        _append_signal_jsonl(out_path, rec)
        state.last_signal_ts = ts.isoformat()
        state.n_emitted += 1
        # Minimal live execution context for dashboard overlays.
        write_execution_state(
            {
                "symbol": symbol,
                "signal": int(rec["signal"]),
                "sl": rec.get("sl"),
                "ttp": None,
                "tp1": None,
                "tp2": None,
                "mode": "signal_only",
                "ts": rec["ts"],
            }
        )
        log.info("live-signal emitted symbol=%s ts=%s signal=%s file=%s", symbol, rec["ts"], rec["signal"], out_path)

    state.last_poll_ts = _now_utc_iso()
    return state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run live IMBA signal worker from KuCoin Futures 1m candles")
    p.add_argument("--symbol", default=os.getenv("LIVE_SYMBOL", "SOL-USDT"))
    p.add_argument("--renko-box", type=float, default=float(os.getenv("LIVE_RENKO_BOX", "0.1")))
    p.add_argument("--lookback", type=int, default=int(os.getenv("LIVE_IMBA_LOOKBACK", "240")))
    p.add_argument("--sl-abs", type=float, default=float(os.getenv("LIVE_IMBA_SL_ABS", "1.5")))
    p.add_argument("--candles-limit", type=int, default=int(os.getenv("LIVE_CANDLES_LIMIT", "1500")))
    p.add_argument("--poll-sec", type=float, default=float(os.getenv("LIVE_POLL_SEC", "15")))
    p.add_argument("--signals-dir", default=os.getenv("SIGNALS_DIR", "data/signals"))
    p.add_argument("--state-file", default=os.getenv("LIVE_SIGNAL_STATE", "data/live/live_signal_state.json"))
    p.add_argument("--once", action="store_true", help="Run one cycle and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    broker = KucoinFuturesBroker()
    signals_dir = Path(args.signals_dir)
    state_path = Path(args.state_file)
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
