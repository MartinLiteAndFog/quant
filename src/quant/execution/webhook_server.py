# src/quant/execution/webhook_server.py

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
import uvicorn

from quant.execution.dashboard_state import (
    build_combined_equity,
    build_equity_curve,
    build_fibo_levels,
    build_trading_diary,
    build_regime_overlay,
    build_regime_scores,
    load_active_levels,
    load_fills_cache_rows,
    load_latest_expected_entry,
    load_live_fill_markers,
    load_kraken_equity_history,
    load_kraken_metrics,
    load_real_equity_history,
    load_renko_bars,
    load_renko_health,
    load_trade_segments,
    load_trade_markers,
)
from quant.execution.gate_provider import get_live_gate_state
from quant.execution.dashboard_statespace import (
    load_state_space_trajectory,
    compute_recent_density,
    refresh_state_space_cache,
)
from quant.regime import RegimeStore, get_live_gate_confidence
from ..utils.log import get_logger, log_throttled

log = get_logger("quant.webhook")
_STATUS_CACHE: Dict[str, Dict[str, Any]] = {}
_POSITION_CACHE: Dict[str, Dict[str, Any]] = {}


def _state_space_refresh_loop() -> None:
    interval = int(os.getenv("DASHBOARD_SS_REFRESH_SEC", "300"))
    while True:
        try:
            info = refresh_state_space_cache()
            if info.get("ok"):
                log_throttled(
                    log,
                    logging.INFO,
                    "webhook_state_space_refresh_ok",
                    float(os.getenv("DASHBOARD_LOG_THROTTLE_SEC", "60")),
                    "state space refresh: %d rows",
                    info.get("rows", 0),
                )
        except Exception as e:
            log_throttled(
                log,
                logging.WARNING,
                "webhook_state_space_refresh_fail",
                float(os.getenv("DASHBOARD_LOG_THROTTLE_SEC", "60")),
                "state space refresh failed: %s",
                e,
            )
        time.sleep(max(60, interval))


@asynccontextmanager
async def _lifespan(a: FastAPI):
    t = threading.Thread(target=_state_space_refresh_loop, daemon=True, name="ss-refresh")
    t.start()
    log.info("state space refresh thread started (interval=%ss)", os.getenv("DASHBOARD_SS_REFRESH_SEC", "300"))
    _start_renko_cache_updater_if_enabled()
    yield


app = FastAPI(title="quant-webhook", version="0.1.0", lifespan=_lifespan)

# Default symbol for dashboard ticker/position
DEFAULT_SYMBOL = os.getenv("DASHBOARD_SYMBOL", "SOL-USDT")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _today_utc() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y%m%d")


def _now_utc_iso() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _symbol_from_payload(payload: Dict[str, Any]) -> str:
    for k in ("symbol", "ticker", "pair"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "UNKNOWN"


def _normalize_symbol(sym: str) -> str:
    s = sym.strip().replace("/", "-").replace(":", "-").replace(" ", "")
    return s or "UNKNOWN"


def _signals_root() -> Path:
    return Path(os.getenv("SIGNALS_DIR", "data/signals"))


def _canon_symbol(sym: str) -> str:
    s = (sym or "").upper()
    return "".join(ch for ch in s if ch.isalnum())


def _norm_symbol_dir(sym: str) -> str:
    return sym.strip().upper().replace("/", "-").replace(":", "-").replace(" ", "")


def _safe_ts(v: Any) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(v, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _latest_signal_from_jsonl(root: Path, symbol: str) -> Optional[Dict[str, Any]]:
    """Return newest non-zero signal from JSONL files (mirrors live_executor._latest_signal)."""
    wanted = _canon_symbol(symbol)
    candidate_dirs: list[Path] = []
    if root.exists():
        for p in root.iterdir():
            if p.is_dir() and _canon_symbol(p.name) == wanted:
                candidate_dirs.append(p)

    if not candidate_dirs:
        sym_dir = root / _norm_symbol_dir(symbol)
        if sym_dir.exists():
            candidate_dirs = [sym_dir]
    if not candidate_dirs:
        return None

    all_files: list[Path] = []
    for d in candidate_dirs:
        all_files.extend(d.glob("*.jsonl"))
    for fp in reversed(sorted(all_files)):
        try:
            with fp.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            for ln in reversed(lines):
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                sig = obj.get("signal")
                ts = _safe_ts(obj.get("ts"))
                if ts is None:
                    continue
                try:
                    sig_i = int(sig)
                except Exception:
                    continue
                if sig_i == 0:
                    continue
                return {"ts": ts, "signal": 1 if sig_i > 0 else -1, "raw": obj}
        except Exception:
            continue
    return None


def _auth_required() -> bool:
    return bool(os.getenv("WEBHOOK_TOKEN", "").strip())


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _cache_ttl_sec() -> float:
    try:
        return float(os.getenv("DASHBOARD_API_CACHE_SEC", "8"))
    except Exception:
        return 8.0


def _cache_get(cache: Dict[str, Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    e = cache.get(key)
    if not isinstance(e, dict):
        return None
    t = float(e.get("_ts", 0.0) or 0.0)
    if (time.time() - t) > max(0.5, _cache_ttl_sec()):
        return None
    v = e.get("value")
    if isinstance(v, dict):
        return dict(v)
    return None


def _cache_put(cache: Dict[str, Dict[str, Any]], key: str, value: Dict[str, Any]) -> None:
    cache[key] = {"_ts": time.time(), "value": dict(value)}


def _start_renko_cache_updater_if_enabled() -> None:
    """
    Optional background updater for dashboard Renko cache.
    Controlled via env:
      ENABLE_DASHBOARD_RENKO_UPDATER=1
    """
    if not _truthy(os.getenv("ENABLE_DASHBOARD_RENKO_UPDATER", "1")):
        return

    symbol = os.getenv("DASHBOARD_SYMBOL", "SOL-USDT")
    out_parquet = os.getenv("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet")
    box = float(os.getenv("DASHBOARD_RENKO_BOX", "0.1"))
    days_back = int(os.getenv("DASHBOARD_RENKO_DAYS_BACK", "14"))
    step_hours = int(os.getenv("DASHBOARD_RENKO_STEP_HOURS", "6"))
    poll_sec = float(os.getenv("DASHBOARD_RENKO_POLL_SEC", "60"))

    def _loop() -> None:
        # Lazy import to avoid startup dependency when updater is disabled.
        from quant.execution.renko_cache_updater import refresh_renko_cache

        while True:
            try:
                info = refresh_renko_cache(
                    symbol=str(symbol),
                    box=float(box),
                    days_back=int(days_back),
                    step_hours=int(step_hours),
                    out_parquet=str(out_parquet),
                )
                log_throttled(
                    log,
                    logging.INFO,
                    "webhook_renko_updater_ok",
                    float(os.getenv("DASHBOARD_LOG_THROTTLE_SEC", "60")),
                    "dashboard renko updater: %s",
                    info,
                )
            except Exception as e:
                log_throttled(
                    log,
                    logging.WARNING,
                    "webhook_renko_updater_fail",
                    float(os.getenv("DASHBOARD_LOG_THROTTLE_SEC", "60")),
                    "dashboard renko updater failed: %s",
                    e,
                )
            time.sleep(max(5.0, float(poll_sec)))

    t = threading.Thread(target=_loop, name="dashboard-renko-updater", daemon=True)
    t.start()
    log.info(
        "started dashboard renko updater symbol=%s out=%s box=%s days_back=%s step_hours=%s poll_sec=%s",
        symbol,
        out_parquet,
        box,
        days_back,
        step_hours,
        poll_sec,
    )


def _start_live_signal_worker_if_enabled() -> None:
    """
    Background thread that computes IMBA signals from live renko bars.
    Controlled via env: ENABLE_LIVE_SIGNAL_WORKER=1
    """
    if not _truthy(os.getenv("ENABLE_LIVE_SIGNAL_WORKER", "1")):
        return

    symbol = os.getenv("LIVE_SYMBOL", "SOL-USDT")
    renko_box = float(os.getenv("LIVE_RENKO_BOX", "0.1"))
    lookback = int(os.getenv("LIVE_IMBA_LOOKBACK", "250"))
    sl_abs = float(os.getenv("LIVE_IMBA_SL_ABS", "1.5"))
    candles_limit = int(os.getenv("LIVE_CANDLES_LIMIT", "1500"))
    poll_sec = float(os.getenv("LIVE_SIGNAL_POLL_SEC", "15"))
    signals_dir = Path(os.getenv("SIGNALS_DIR", "data/signals"))
    state_file = Path(os.getenv("LIVE_SIGNAL_STATE", "data/live/live_signal_state.json"))
    default_gate_on = int(os.getenv("LIVE_DEFAULT_GATE_ON", "0"))

    def _loop() -> None:
        from quant.execution.live_signal_worker import run_once as sw_run_once, WorkerState, _read_state, _write_state
        from quant.brokers.kucoin_futures import KucoinFuturesBroker
        from quant.regime.store import RegimeStore

        broker = KucoinFuturesBroker()
        regime_store = RegimeStore()
        st = _read_state(state_file)

        while True:
            try:
                st = sw_run_once(
                    broker, symbol=symbol, renko_box=renko_box, lookback=lookback,
                    sl_abs=sl_abs, candles_limit=candles_limit,
                    signals_dir=signals_dir, regime_store=regime_store,
                    default_gate_on=default_gate_on, state=st,
                )
                _write_state(state_file, st)
            except Exception as e:
                log.warning("live signal worker error: %s", e)
            time.sleep(max(5.0, poll_sec))

    t = threading.Thread(target=_loop, name="live-signal-worker", daemon=True)
    t.start()
    log.info(
        "started live signal worker symbol=%s box=%s lookback=%s poll_sec=%s",
        symbol, renko_box, lookback, poll_sec,
    )


def _start_live_executor_if_enabled() -> None:
    """
    Background thread that processes signals and places trades via OMS.
    Controlled via env: ENABLE_LIVE_EXECUTOR=1 (default 0 for safety).
    """
    if not _truthy(os.getenv("ENABLE_LIVE_EXECUTOR", "0")):
        return

    symbol = os.getenv("LIVE_SYMBOL", "SOL-USDT")
    signals_dir = Path(os.getenv("SIGNALS_DIR", "data/signals"))
    state_file = Path(os.getenv("LIVE_EXECUTOR_STATE", "data/live/live_executor_state.json"))
    poll_sec = float(os.getenv("LIVE_EXECUTOR_POLL_SEC", "5"))
    live_enabled = _truthy(os.getenv("LIVE_TRADING_ENABLED", "0"))
    dry_run = _truthy(os.getenv("LIVE_EXECUTOR_DRY_RUN", "1"))
    max_eur = float(os.getenv("LIVE_EXECUTOR_MAX_EUR", "20"))
    leverage = float(os.getenv("LIVE_EXECUTOR_LEVERAGE", "1"))

    allowlist_raw = os.getenv("LIVE_EXECUTOR_SYMBOL_ALLOWLIST", "SOL-USDT")
    allowlist = {s.strip().upper() for s in allowlist_raw.split(",") if s.strip()}
    sym_upper = symbol.strip().upper()
    if sym_upper not in allowlist:
        log.warning("live executor: symbol %s not in allowlist %s – not starting", sym_upper, allowlist)
        return

    def _loop() -> None:
        from quant.execution.live_executor import run_once as ex_run_once, ExecutorState, _read_state, _write_state
        from quant.execution.kucoin_futures import KucoinFuturesBroker
        from quant.execution.oms import MakerFirstOMS, OmsDefaults

        broker = KucoinFuturesBroker()
        oms = MakerFirstOMS(broker=broker, cfg=OmsDefaults())
        st = _read_state(state_file)

        while True:
            try:
                st = ex_run_once(
                    broker=broker, oms=oms, symbol=symbol,
                    signals_root=signals_dir, state=st,
                    live_enabled=live_enabled, dry_run=dry_run,
                    max_eur=max_eur, leverage=leverage,
                )
                _write_state(state_file, st)
            except Exception as e:
                log.warning("live executor error: %s", e)
            time.sleep(max(1.0, poll_sec))

    t = threading.Thread(target=_loop, name="live-executor", daemon=True)
    t.start()
    log.info(
        "started live executor symbol=%s live_enabled=%s dry_run=%s max_eur=%s leverage=%s poll_sec=%s",
        symbol, live_enabled, dry_run, max_eur, leverage, poll_sec,
    )


def _sync_gate_conf_artifacts_if_enabled() -> None:
    """
    Optional one-way sync of gate-confidence files from app workspace -> mounted volume.
    This is useful when Railway volume does not yet contain required state-space artifacts.
    """
    if not _truthy(os.getenv("GATE_CONF_SYNC_ON_START", "0")):
        return

    src_dir = Path(os.getenv("GATE_CONF_SYNC_SRC_DIR", "data/runs/visual_v02_seed/transitions"))
    src_gate = Path(
        os.getenv(
            "GATE_CONF_SYNC_SRC_GATE",
            "data/regimes/SOLUSDT_tv5mIMBA_gate2of3_qch0.4_qadx0.6_qer0.3_daily.csv",
        )
    )
    dst_dir = Path(os.getenv("GATE_CONF_ARTIFACT_DIR", "/data/live/gate_conf/transitions"))
    dst_gate = Path(os.getenv("GATE_DAILY_PATH", "/data/live/gate_conf/gate_daily.csv"))

    required = [
        "voxel_map.parquet",
        "voxel_stats.parquet",
        "transitions_topk.parquet",
        "basins_v02_components.parquet",
    ]
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        if dst_gate.parent:
            dst_gate.parent.mkdir(parents=True, exist_ok=True)

        copied = []
        for name in required:
            s = src_dir / name
            d = dst_dir / name
            if not s.exists():
                log.warning("gate-conf sync missing source file: %s", s)
                continue
            need_copy = (not d.exists()) or (s.stat().st_mtime > d.stat().st_mtime + 1e-6) or (s.stat().st_size != d.stat().st_size)
            if need_copy:
                shutil.copy2(s, d)
                copied.append(str(d))

        if src_gate.exists():
            need_copy_gate = (not dst_gate.exists()) or (src_gate.stat().st_mtime > dst_gate.stat().st_mtime + 1e-6) or (src_gate.stat().st_size != dst_gate.stat().st_size)
            if need_copy_gate:
                shutil.copy2(src_gate, dst_gate)
                copied.append(str(dst_gate))
        else:
            log.warning("gate-conf sync missing source gate file: %s", src_gate)

        if copied:
            log.info("gate-conf sync copied %d files", len(copied))
        else:
            log.info("gate-conf sync up-to-date; no files copied")
    except Exception as e:
        log.warning("gate-conf sync failed: %s", e)


def _check_token(token: Optional[str]) -> None:
    expected = os.getenv("WEBHOOK_TOKEN", "").strip()
    if not expected:
        return
    if not token or token.strip() != expected:
        raise HTTPException(status_code=401, detail="invalid webhook token")


def _ensure_ts(payload: Dict[str, Any], now_iso: str) -> Dict[str, Any]:
    """
    Guarantee a 'ts' field exists for downstream backtests.
    - If client sends ts/timestamp/time/t/datetime -> normalize into 'ts'
    - Else fallback to server_ts (now)
    Keep original fields too.
    """
    candidate_keys = ("ts", "timestamp", "time", "t", "datetime")
    ts_val = None
    for k in candidate_keys:
        if k in payload and payload[k] is not None and str(payload[k]).strip():
            ts_val = payload[k]
            break

    out = dict(payload)
    if ts_val is None:
        out["ts"] = now_iso
        out["_ts_source"] = "server_ts_fallback"
    else:
        # preserve exact value under 'ts' for signal_io
        out["ts"] = ts_val
        out["_ts_source"] = f"payload:{k}"
    return out


@app.get("/")
def root() -> Dict[str, Any]:
    """Root für Health-Checks (z. B. Railway); leitet auf Dashboard hin."""
    return {"ok": True, "app": "quant-webhook", "ts": _now_utc_iso(), "dashboard": "/dashboard", "health": "/health"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": _now_utc_iso()}


def _kucoin_broker():
    """Lazy import so app starts even without credentials."""
    from quant.execution.kucoin_futures import KucoinFuturesBroker
    return KucoinFuturesBroker()


@app.get("/api/status")
def api_status(symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """API status: whether KuCoin credentials are set, and current ticker (bid/ask) from KuCoin."""
    cache_key = _normalize_symbol(symbol)
    cached = _cache_get(_STATUS_CACHE, cache_key)
    if cached is not None:
        return cached

    key = (os.getenv("KUCOIN_FUTURES_API_KEY") or "").strip()
    out = {"ok": True, "ts": _now_utc_iso(), "api_configured": bool(key)}
    if not key:
        out["ticker"] = None
        out["balance"] = None
        out["hint"] = "Set KUCOIN_FUTURES_API_KEY, KUCOIN_FUTURES_API_SECRET, KUCOIN_FUTURES_PASSPHRASE (e.g. in .env or cloud env vars)."
        _cache_put(_STATUS_CACHE, cache_key, out)
        return out
    try:
        broker = _kucoin_broker()
        bid, ask = broker.get_best_bid_ask(symbol)
        out["ticker"] = {"symbol": symbol, "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0 if (bid and ask) else None}
    except Exception as e:
        out["ticker"] = None
        out["ticker_error"] = str(e)
    try:
        broker = _kucoin_broker()
        out["balance"] = broker.get_account_balance("USDT")
    except Exception as e:
        out["balance"] = None
        out["balance_error"] = str(e)
    _cache_put(_STATUS_CACHE, cache_key, out)
    return out


@app.get("/api/position")
def api_position(symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """Current position from KuCoin Futures (signed: >0 long, <0 short)."""
    cache_key = _normalize_symbol(symbol)
    cached = _cache_get(_POSITION_CACHE, cache_key)
    if cached is not None:
        return cached

    key = (os.getenv("KUCOIN_FUTURES_API_KEY") or "").strip()
    if not key:
        out = {"ok": True, "symbol": symbol, "position": None, "side": None, "leverage": None, "hint": "Configure KuCoin API keys."}
        _cache_put(_POSITION_CACHE, cache_key, out)
        return out
    try:
        broker = _kucoin_broker()
        info = broker.get_position_info(symbol)
        out = {
            "ok": True,
            "symbol": symbol,
            "position": info.get("size"),
            "side": info.get("side"),
            "leverage": info.get("leverage"),
        }
        _cache_put(_POSITION_CACHE, cache_key, out)
        return out
    except Exception as e:
        out = {"ok": False, "symbol": symbol, "position": None, "side": None, "leverage": None, "error": str(e)}
        _cache_put(_POSITION_CACHE, cache_key, out)
        return out


@app.post("/api/manual/order")
async def api_manual_order(
    request: Request,
    x_webhook_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """
    Execute a manual order using the same KuCoin adapter as live execution.
    Example payload:
      {"symbol":"SOL-USDT","action":"cancel_short"}
    """
    if _auth_required():
        _check_token(x_webhook_token)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")

    symbol = str(payload.get("symbol") or DEFAULT_SYMBOL)
    action = str(payload.get("action") or "").strip()
    if not action:
        raise HTTPException(status_code=400, detail="missing action")

    qty_raw = payload.get("qty")
    qty = None
    if qty_raw not in (None, ""):
        try:
            qty = float(qty_raw)
        except Exception:
            raise HTTPException(status_code=400, detail="qty must be numeric")

    try:
        wait_sec = float(payload.get("wait_sec", os.getenv("MANUAL_ORDER_WAIT_SEC", "20")))
    except Exception:
        wait_sec = 20.0
    dry_run = bool(payload.get("dry_run", False))

    try:
        from quant.execution.manual_orders import execute_manual_action

        result = execute_manual_action(
            broker=_kucoin_broker(),
            symbol=symbol,
            action=action,
            qty=qty,
            wait_sec=wait_sec,
            dry_run=dry_run,
        )
        if isinstance(result, dict):
            result.setdefault("ts", _now_utc_iso())
            return result
        return {"ok": True, "result": result, "ts": _now_utc_iso()}
    except Exception as e:
        return {"ok": False, "symbol": symbol, "action": action, "error": str(e), "ts": _now_utc_iso()}


@app.get("/api/regime/latest")
def api_regime_latest(symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    try:
        row = RegimeStore().get_latest_state(symbol=symbol)
        return {"ok": True, "symbol": symbol, "regime": row}
    except Exception as e:
        return {"ok": False, "symbol": symbol, "regime": None, "error": str(e)}


@app.get("/api/regime/transitions")
def api_regime_transitions(symbol: str = DEFAULT_SYMBOL, limit: int = 50) -> Dict[str, Any]:
    try:
        rows = RegimeStore().get_recent_transitions(symbol=symbol, limit=int(max(1, limit)))
        return {"ok": True, "symbol": symbol, "transitions": rows}
    except Exception as e:
        return {"ok": False, "symbol": symbol, "transitions": [], "error": str(e)}


@app.get("/api/dashboard/chart")
def api_dashboard_chart(
    symbol: str = DEFAULT_SYMBOL,
    hours: int = 24 * 7,
    max_points: int = 3000,
) -> Dict[str, Any]:
    """
    Unified chart payload: renko bars, trades, regime overlays, and active levels.
    """
    try:
        bars = load_renko_bars(max_points=int(max(100, max_points)))
        markers = load_trade_markers(max_points=int(max(1000, max_points * 50)))
        oldest_bar_ts = int(bars[0]["time"]) if bars else None
        markers_live = load_live_fill_markers(
            symbol=symbol,
            start_ts=oldest_bar_ts,
            limit=int(max(200, min(int(os.getenv("DASHBOARD_FILL_MARKER_LIMIT", "1200")), max_points))),
      )
        levels = load_active_levels()
        expected_entry ={}

        def _coerce_epoch_seconds(v: Any) -> Optional[int]:
            if v is None:
                return None
            if isinstance(v, pd.Timestamp):
                return int(pd.Timestamp(v).timestamp())
            if isinstance(v, (int, float)):
                try:
                    x = float(v)
                except Exception:
                    return None
                if not (x > 0):
                    return None
                # Heuristic for ns/ms/s
                if x > 10**15:
                    return int(x / 1e9)   # ns -> s
                if x > 10**12:
                    return int(x / 1e3)   # ms -> s
                return int(x)             # s
            s = str(v).strip()
            if not s:
                return None
            try:
                return _coerce_epoch_seconds(float(s))
            except Exception:
                pass
            try:
                ts = pd.to_datetime(s, utc=True, errors="coerce")
                if pd.isna(ts):
                    return None
                return int(pd.Timestamp(ts).timestamp())
            except Exception:
                return None

        # Normalize common timestamp fields so the frontend can Number(...) them reliably.
        if isinstance(levels, dict) and levels:
            for k in ("entry_bar_ts", "ts"):
                if k in levels:
                    t_norm = _coerce_epoch_seconds(levels.get(k))
                    if t_norm is not None:
                        levels[k] = t_norm

        def _side_to_int(v: Any) -> int:
            if v is None:
                return 0
            if isinstance(v, (int, float)):
                try:
                    return int(v)
                except Exception:
                    return 0
            s = str(v).strip().lower()
            if s in ("long", "l", "buy"):
                return 1
            if s in ("short", "s", "sell"):
                return -1
            try:
                return int(float(s))
            except Exception:
                return 0

        live_entry_marker = None
        try:
            entry_t = levels.get("entry_bar_ts")
            entry_px = levels.get("entry_px")
            side_raw = levels.get("side")
            if side_raw is None:
                side_raw = expected_entry.get("side")
            if (entry_t is None) and isinstance(levels, dict):
                # Fallback: signal-only state may carry a 'ts' timestamp (ISO or epoch).
                entry_t = levels.get("ts")
            if entry_t is None:
                entry_t = expected_entry.get("entry_time")
            if entry_px is None:
                entry_px = expected_entry.get("entry_price")
            if entry_t is not None and side_raw is not None:
                t_i = _coerce_epoch_seconds(entry_t)
                if t_i is None:
                    t_i = 0
                side_i = _side_to_int(side_raw)
                px_f = float(entry_px) if entry_px is not None and str(entry_px).strip() else None
                if t_i > 0 and side_i != 0:
                    live_entry_marker = {
                        "time": t_i,
                        "position": "belowBar" if side_i >= 0 else "aboveBar",
                        "shape": "arrowUp" if side_i >= 0 else "arrowDown",
                        "color": "#7aa2f7",
                        "text": f"live entry {'L' if side_i >= 0 else 'S'}" + (f" @ {px_f:.3f}" if px_f else ""),
                    }
        except Exception:
            live_entry_marker = None

        markers_all = markers + markers_live
        if live_entry_marker is not None:
            mt = int(live_entry_marker.get("time", 0))
            dup = any(int(m.get("time", 0)) == mt and str(m.get("shape", "")) == str(live_entry_marker.get("shape")) for m in markers_all)
            if not dup:
                markers_all.append(live_entry_marker)

        markers = sorted(markers_all, key=lambda x: int(x.get("time", 0)))
        if len(markers) > int(max(100, max_points)):
            markers = markers[-int(max(100, max_points)):]
        segments = load_trade_segments(max_points=int(max(100, max_points)))
        fibo = build_fibo_levels(max_points=int(max(100, max_points)))
        renko_health = load_renko_health()
        regime = build_regime_overlay(symbol=symbol, hours=int(max(1, hours)))
        latest = regime.get("latest") or {}
        live_gc = None
        live_gc_error = "temporarily_disabled"
        selected_p_trend = (live_gc or {}).get("selected_p_trend")
        live_conf = float(max(0.0, min(1.0, selected_p_trend))) if isinstance(selected_p_trend, (float, int)) else None
        if live_conf is not None and isinstance(regime.get("latest"), dict):
            regime["latest"]["confidence"] = live_conf
        if live_conf is not None and isinstance(regime.get("spans"), list) and regime["spans"]:
            regime["spans"][-1]["confidence"] = live_conf
        confidence_out = live_conf if live_conf is not None else latest.get("confidence")

        regime_score_data = build_regime_scores(symbol=symbol, hours=int(max(1, hours)))
        equity = build_equity_curve(max_points=int(max(100, max_points)))
        equity_real = load_real_equity_history(max_points=int(max(100, max_points)))
        equity_kraken = load_kraken_equity_history(max_points=int(max(100, max_points)))
        kraken_metrics = load_kraken_metrics()
        equity_combined = build_combined_equity(
            kucoin_points=equity_real.get("points", []),
            kraken_points_usd=equity_kraken.get("points", []),
        )
        diary = build_trading_diary(max_points=int(max(100, max_points)))

        open_position = None
        try:
            side_raw = levels.get("side")
            entry_t = levels.get("entry_bar_ts")
            entry_px = levels.get("entry_px")
            sl = levels.get("sl")
            mode = levels.get("mode")
            if side_raw is None:
                side_raw = expected_entry.get("side")
            side_i = _side_to_int(side_raw)
            if (entry_t is None) and isinstance(levels, dict):
                entry_t = levels.get("ts")
            if entry_t is None:
                entry_t = expected_entry.get("entry_time")
            if entry_px is None:
                entry_px = expected_entry.get("entry_price")
            t_i = _coerce_epoch_seconds(entry_t) if entry_t is not None and str(entry_t).strip() else None
            px_f = float(entry_px) if entry_px is not None and str(entry_px).strip() else None
            if side_i != 0 and t_i and t_i > 0:
                open_position = {
                    "side": "long" if side_i >= 0 else "short",
                    "entry_time": t_i,
                    "entry_price": px_f,
                    "sl": float(sl) if sl is not None and str(sl).strip() else None,
                    "mode": str(mode) if mode is not None else None,
                }
        except Exception:
            open_position = None

        regime_forecast: list[dict[str, Any]] = []
        if live_gc and isinstance(live_gc.get("horizons"), list):
            now_ts = pd.Timestamp.now("UTC")
            for h in live_gc["horizons"]:
                minutes = h.get("minutes", 0)
                p_trend = h.get("p_trend_voxel")
                if p_trend is not None and isinstance(p_trend, (int, float)):
                    score = round(2.0 * float(p_trend) - 1.0, 4)
                    forecast_ts = int((now_ts + pd.Timedelta(minutes=minutes)).timestamp())
                    regime_forecast.append({"time": forecast_ts, "score": score})

        # #region agent log
        _newest_bar_ts = int(bars[-1]["time"]) if bars else None
        _marker_times = sorted([int(m.get("time", 0)) for m in markers])
        _debug = {
            "renko_bars_count": len(bars),
            "oldest_bar_ts": oldest_bar_ts,
            "newest_bar_ts": _newest_bar_ts,
            "markers_from_trades_parquet": len(markers),
            "markers_from_live_fills": len(markers_live),
            "markers_total_after_merge": len(markers),
            "live_entry_marker": live_entry_marker,
            "marker_newest_5_times": _marker_times[-5:] if _marker_times else [],
            "levels_keys": sorted(levels.keys()) if isinstance(levels, dict) else None,
            "levels_side": levels.get("side") if isinstance(levels, dict) else None,
            "levels_entry_bar_ts": levels.get("entry_bar_ts") if isinstance(levels, dict) else None,
            "levels_entry_px": levels.get("entry_px") if isinstance(levels, dict) else None,
            "levels_ts": levels.get("ts") if isinstance(levels, dict) else None,
            "expected_entry": expected_entry if expected_entry else None,
            "open_position": open_position,
            "diary_count": len(diary.get("entries", [])),
            "diary_source": diary.get("source"),
            "equity_count": len(equity.get("trades", [])),
        }
        # #endregion

        return {
            "ok": True,
            "symbol": symbol,
            "bars": bars,
            "markers": markers,
            "levels": levels,
            "ttp_trail_pct": float(levels.get("ttp_trail_pct") or os.getenv("LIVE_FLIP_TTP_TRAIL_PCT", os.getenv("LIVE_TTP_TRAIL_PCT", "0.012"))),
            "regime": regime,
            "confidence": confidence_out,
            "gate_on": latest.get("gate_on"),
            "regime_state": latest.get("regime_state"),
            "gate_confidence": live_gc,
            "gate_confidence_error": live_gc_error,
            "segments": segments,
            "fibo": fibo,
            "renko_health": renko_health,
            "regime_scores": regime_score_data.get("scores", []),
            "regime_forecast": regime_forecast,
            "equity_curve": equity.get("trades", []),
            "equity_source": equity.get("source"),
            "equity_real": equity_real.get("points", []),
            "equity_real_source": equity_real.get("source"),
            "equity_kraken": equity_kraken.get("points", []),
            "equity_kraken_source": equity_kraken.get("source"),
            "equity_combined": equity_combined.get("points", []),
            "equity_combined_source": equity_combined.get("source"),
            "equity_live": equity_kraken.get("points", []),
            "equity_live_source": equity_kraken.get("source"),
            "equity_realized": equity_combined.get("points", []),
            "equity_realized_source": equity_combined.get("source"),
            "kraken_metrics": kraken_metrics,
            "diary_entries": diary.get("entries", []),
            "diary_source": diary.get("source"),
            "open_position": open_position,
            "_debug": _debug,
            "ts": _now_utc_iso(),
        }
    except Exception as e:
        return {
            "ok": False,
            "symbol": symbol,
            "bars": [],
            "markers": [],
            "levels": {},
            "regime": {"spans": [], "points": [], "latest": None},
            "segments": [],
            "fibo": {"lookback": None, "long": [], "mid": [], "short": [], "latest": {}},
            "renko_health": {"ok": False, "bars": 0, "last_ts": None, "age_sec": None},
            "regime_scores": [],
            "regime_forecast": [],
            "equity_curve": [],
            "equity_source": "none",
            "equity_real": [],
            "equity_real_source": "none",
            "equity_kraken": [],
            "equity_kraken_source": "none",
            "equity_combined": [],
            "equity_combined_source": "none",
            "equity_live": [],
            "equity_live_source": "none",
            "equity_realized": [],
            "equity_realized_source": "none",
            "kraken_metrics": {},
            "diary_entries": [],
            "diary_source": "none",
            "open_position": None,
            "error": str(e),
            "ts": _now_utc_iso(),
        }


@app.get("/api/gate/solusd")
def api_gate_solusd() -> Dict[str, Any]:
    try:
        out = get_live_gate_state()
        return {
            "ok": True,
            "ts": out.get("ts"),
            "gate_on": int(out.get("gate_on", 0) or 0),
            "gate_off": int(out.get("gate_off", 1) or 1),
            "source": out.get("source"),
            "x": out.get("x"),
            "y": out.get("y"),
            "z": out.get("z"),
            "g1_drift": out.get("g1_drift"),
            "g2_elasticity": out.get("g2_elasticity"),
            "g3_instability": out.get("g3_instability"),
            "age_sec": out.get("age_sec"),
        }
    except Exception as e:
        return {
            "ok": False,
            "ts": _now_utc_iso(),
            "gate_on": 0,
            "gate_off": 1,
            "source": "error",
            "error": str(e),
        }


@app.get("/api/signals/latest/solusd")
def api_signals_latest_solusd() -> Dict[str, Any]:
    try:
        redis_url = os.getenv("REDIS_URL", "").strip()
        if redis_url:
            import json
            import redis as redis_lib

            r = redis_lib.from_url(redis_url, decode_responses=True)
            raw = r.get("signal:SOLUSDT:latest")
            if raw:
                obj = json.loads(raw)
                return {
                    "ok": True,
                    "ts": str(obj.get("ts", _now_utc_iso())),
                    "signal": int(obj.get("signal", 0) or 0),
                    "source": "redis",
                }

        root = _signals_root()
        sig = _latest_signal_from_jsonl(root, "SOL-USDT")
        if sig is None:
            return {"ok": True, "ts": _now_utc_iso(), "signal": 0, "source": "no_signal"}
        return {
            "ok": True,
            "ts": str(sig["ts"]),
            "signal": int(sig["signal"]),
            "source": "jsonl",
        }
    except Exception as e:
        return {"ok": False, "ts": _now_utc_iso(), "signal": 0, "error": str(e)}

@app.get("/api/renko/latest/solusd")
def api_renko_latest_solusd(lookback: int = 50) -> Dict[str, Any]:
    try:
        from quant.execution.dashboard_state import _refresh_renko_cache_if_needed, _read_renko_df

        path = Path(os.getenv("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet"))
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = pd.DataFrame()
        if df.empty or "close" not in df.columns:
            return {"ok": False, "error": "no_renko_data"}
        df = df.sort_values("ts") if "ts" in df.columns else df
        lb = min(max(lookback, 1), 50)  # backtest caps swing lookback to 50
        swing_low = float(df["low"].rolling(lb, min_periods=1).min().iloc[-1])
        swing_high = float(df["high"].rolling(lb, min_periods=1).max().iloc[-1])
        return {
            "ok": True,
            "ts": str(df["ts"].iloc[-1]) if "ts" in df.columns else _now_utc_iso(),
            "swing_low": round(swing_low, 6),
            "swing_high": round(swing_high, 6),
            "last_close": round(float(df["close"].iloc[-1]), 6),
            "n_bars": len(df),
            "lookback_used": lb,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/dashboard/diary")
def api_dashboard_diary(max_points: int = 500) -> Dict[str, Any]:
    try:
        diary = build_trading_diary(max_points=int(max(10, max_points)))
        return {"ok": True, "entries": diary.get("entries", []), "source": diary.get("source"), "ts": _now_utc_iso()}
    except Exception as e:
        return {"ok": False, "entries": [], "source": "none", "error": str(e), "ts": _now_utc_iso()}


@app.get("/api/dashboard/fills")
def api_dashboard_fills(symbol: str = DEFAULT_SYMBOL, max_points: int = 500) -> Dict[str, Any]:
    try:
        rows = load_fills_cache_rows(symbol=symbol, max_points=int(max(10, max_points)))
        return {"ok": True, "rows": rows, "count": len(rows), "ts": _now_utc_iso()}
    except Exception as e:
        return {"ok": False, "rows": [], "count": 0, "error": str(e), "ts": _now_utc_iso()}


def _load_density_bg_images() -> Dict[str, Optional[str]]:
    """Load pre-computed density PNGs as base64 strings."""
    import base64
    density_dir = Path(os.getenv("DASHBOARD_DENSITY_DIR", "data/live/density"))
    out: Dict[str, Optional[str]] = {}
    for tag in ("xy", "xz", "yz"):
        p = density_dir / f"density_bg_{tag}.png"
        if p.exists():
            data = p.read_bytes()
            out[tag] = f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"
        else:
            out[tag] = None
    return out


@app.get("/api/dashboard/statespace")
def api_dashboard_statespace(window_hours: float = 8.0) -> Dict[str, Any]:
    """State space heatmap data: trajectory, current position, density layers."""
    try:
        traj = load_state_space_trajectory(window_hours=float(max(0.1, window_hours)))
        recent = compute_recent_density(hours=min(window_hours, 12.0))
        density_bg = _load_density_bg_images()
        return {
            "ok": True,
            "trajectory": traj.get("trajectory", []),
            "current": traj.get("current"),
            "recent_density": recent,
            "density_bg": density_bg,
            "window_hours": window_hours,
        }
    except Exception as e:
        return {"ok": False, "trajectory": [], "current": None,
                "recent_density": {"xy": [], "xz": [], "yz": []},
                "density_bg": {"xy": None, "xz": None, "yz": None},
                "error": str(e)}


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quant Live Dashboard</title>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root { --bg: #141823; --card: #1e2333; --text: #d9def7; --muted: #8e98bf; --ok: #9ece6a; --err: #f7768e; --accent: #7aa2f7; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 1rem; font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); }
    h1 { margin: 0 0 0.75rem 0; color: var(--accent); font-size: 1.25rem; }
    .layout { display: grid; grid-template-columns: 1fr 320px; gap: 1rem; align-items: start; }
    .card { background: var(--card); border-radius: 10px; padding: 0.75rem; }
    .chart-wrap { position: relative; height: 620px; border-radius: 10px; overflow: hidden; }
    #chart { position: absolute; inset: 0; }
    .chart-refresh-btn {
      position: absolute;
      top: 8px;
      right: 8px;
      z-index: 4;
      background: rgba(20,24,35,0.92);
      color: var(--text);
      border: 1px solid #3a4b72;
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 0.74rem;
      cursor: pointer;
    }
    .chart-refresh-btn:disabled { opacity: 0.65; cursor: default; }
    .row { display: flex; justify-content: space-between; gap: 1rem; margin: 0.4rem 0; }
    .label { color: var(--muted); }
    .ok { color: var(--ok); }
    .err { color: var(--err); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .hint { color: var(--muted); font-size: 0.85rem; margin-top: 0.5rem; }
    .confidence-pill { font-weight: 700; }
    .regime-band { height: 35px; border-radius: 6px; overflow: hidden; position: relative; }
    .regime-band canvas { width: 100%; height: 100%; display: block; }
    .bottom-row { display: grid; grid-template-columns: 1fr 1fr 1fr 280px; gap: 1rem; margin-top: 1rem; }
    .heatmap-card { position: relative; }
    .heatmap-card canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 6px; }
    .heatmap-title { color: var(--text); font-size: 0.85rem; text-align: center; margin-bottom: 0.25rem; }
    .traj-controls { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.75rem; }
    .traj-controls select { background: var(--card); color: var(--text); border: 1px solid #2a3044; border-radius: 4px; padding: 2px 6px; font-size: 0.85rem; }
    .fills-list { margin-top: 0.35rem; max-height: 190px; overflow: auto; border: 1px solid #2a3044; border-radius: 6px; }
    .fills-row { display: grid; grid-template-columns: 122px 40px 40px 56px 1fr; gap: 6px; padding: 3px 6px; font-size: 0.72rem; border-bottom: 1px solid #252b3f; }
    .fills-row:last-child { border-bottom: 0; }
    .fills-row.head { color: var(--muted); font-weight: 600; position: sticky; top: 0; background: #1e2333; z-index: 1; }
    .fills-buy { color: #2ecc71; }
    .fills-sell { color: #f7768e; }
    .manual-grid { display: grid; grid-template-columns: 1fr 72px; gap: 6px; margin-top: 0.35rem; }
    .manual-grid select, .manual-grid input, .manual-grid button {
      background: #141823; color: var(--text); border: 1px solid #2a3044; border-radius: 6px; padding: 4px 6px; font-size: 0.75rem;
    }
    .manual-btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 0.35rem; }
    .manual-btn-row button {
      background: #1a2030; color: var(--text); border: 1px solid #2a3044; border-radius: 6px; padding: 4px 6px; font-size: 0.74rem; cursor: pointer;
    }
    .manual-send { width: 100%; margin-top: 0.35rem; background: #25314a; color: var(--text); border: 1px solid #3a4b72; border-radius: 6px; padding: 5px 8px; font-size: 0.78rem; cursor: pointer; }
    .manual-result { font-size: 0.72rem; margin-top: 0.35rem; min-height: 1rem; white-space: pre-wrap; word-break: break-word; }
    @media (max-width: 1200px) { .layout { grid-template-columns: 1fr; } .chart-wrap { height: 520px; } .bottom-row { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 800px) { .bottom-row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>Quant Live Dashboard</h1>
  <div class="layout">
    <div class="card">
      <div class="chart-wrap">
        <div id="chart"></div>
      </div>
    </div>
    <div class="card">
      <div class="row"><span class="label">API (KuCoin)</span><span id="api-status">...</span></div>
      <div class="row"><span class="label">Ticker</span><span id="ticker" class="mono">...</span></div>
      <div class="row"><span class="label">Position</span><span id="position" class="mono">...</span></div>
      <div class="row"><span class="label">Notional (est)</span><span id="position-notional" class="mono">...</span></div>
      <hr style="border-color:#2a3044;border-style:solid;border-width:1px 0 0 0;margin:0.8rem 0;">
      <div class="row"><span class="label">Capital</span><span id="capital" class="mono ok">...</span></div>
      <div class="row"><span class="label">Regime</span><span id="regime-state">...</span></div>
      <div class="row"><span class="label">Confidence</span><span id="confidence" class="confidence-pill">...</span></div>
      <div class="row"><span class="label">Bar time</span><span id="bar-time" class="mono">-</span></div>
      <div class="row"><span class="label">Exit mode</span><span id="exit-mode">-</span></div>
      <div class="row"><span class="label">SL</span><span id="lvl-sl" class="mono">-</span></div>
      <div class="row"><span class="label">TTP</span><span id="lvl-ttp" class="mono">-</span></div>
      <div class="row"><span class="label">TP1</span><span id="lvl-tp1" class="mono">-</span></div>
      <div class="row"><span class="label">TP2</span><span id="lvl-tp2" class="mono">-</span></div>
      <div style="margin-top:0.5rem;font-size:0.8rem;color:var(--muted);font-weight:600;">Equity Curve</div>
      <canvas id="equity-canvas" style="width:100%;height:160px;display:block;border-radius:6px;margin-top:0.25rem;"></canvas>
      <div id="equity-meta" class="hint" style="margin-top:0.3rem;">Diary source: -</div>
      <div id="equity-detail" class="mono" style="font-size:0.75rem;color:var(--text);min-height:1.1rem;">-</div>
      <div style="margin-top:0.5rem;font-size:0.8rem;color:var(--muted);font-weight:600;">Recent fills (raw)</div>
      <div id="fills-list" class="fills-list">
        <div class="fills-row head"><span>time (UTC)</span><span>side</span><span>qty</span><span>price</span><span>reason</span></div>
      </div>
      <div style="margin-top:0.55rem;font-size:0.8rem;color:var(--muted);font-weight:600;">Manual orders</div>
      <div class="manual-grid">
        <select id="manual-action" class="mono">
          <option value="cancel_short" selected>cancel_short</option>
          <option value="cancel_long">cancel_long</option>
          <option value="flatten">flatten</option>
          <option value="enter_long">enter_long</option>
          <option value="enter_short">enter_short</option>
          <option value="cancel_all_orders">cancel_all_orders</option>
        </select>
        <input id="manual-qty" class="mono" type="number" min="0" step="1" placeholder="qty">
      </div>
      <div class="manual-btn-row">
        <button id="manual-cancel-short" type="button">Cancel short</button>
        <button id="manual-cancel-long" type="button">Cancel long</button>
      </div>
      <button id="manual-send" class="manual-send" type="button">Send manual order</button>
      <div id="manual-result" class="manual-result mono">-</div>
      <p id="hint" class="hint"></p>
    </div>
  </div>

  <div class="card regime-band" style="margin-top:0.5rem;grid-column:1;">
    <canvas id="regime-canvas"></canvas>
  </div>
  <div class="hint" style="text-align:center;margin-top:0.25rem;">Regime: red = countertrend, green = trend. Right side = projected.</div>

  <div class="traj-controls">
    <span class="label" style="font-size:0.85rem;">Chart range:</span>
    <select id="chart-range" class="mono">
      <option value="14d" selected>14d</option>
      <option value="30d">30d</option>
      <option value="all">all</option>
    </select>
    <span class="label" style="font-size:0.85rem;">Trajectory window:</span>
    <select id="traj-window" class="mono">
      <option value="1">1h</option>
      <option value="4">4h</option>
      <option value="8" selected>8h</option>
      <option value="12">12h</option>
      <option value="24">24h</option>
      <option value="48">48h</option>
    </select>
    <span class="label" style="font-size:0.85rem;margin-left:1.5rem;">Time cursor:</span>
    <input type="range" id="traj-slider" min="0" max="100" value="100" style="width:200px;accent-color:var(--accent);">
    <span id="traj-slider-label" class="mono" style="font-size:0.8rem;min-width:40px;">now</span>
  </div>

  <div class="bottom-row">
    <div class="card heatmap-card">
      <div class="heatmap-title">Drift vs Elasticity</div>
      <canvas id="heatmap-xy"></canvas>
    </div>
    <div class="card heatmap-card">
      <div class="heatmap-title">Drift vs Instability</div>
      <canvas id="heatmap-xz"></canvas>
    </div>
    <div class="card heatmap-card">
      <div class="heatmap-title">Elasticity vs Instability</div>
      <canvas id="heatmap-yz"></canvas>
    </div>
    <div class="card" id="axis-bars-card">
      <div style="color:var(--text);font-size:0.95rem;font-weight:600;margin-bottom:0.5rem;">Current State</div>
      <canvas id="axis-bars" width="250" height="200"></canvas>
    </div>
  </div>

  <script>
    const chartEl = document.getElementById('chart');
    const qs = new URLSearchParams(window.location.search);
    const uiRefreshMsDefault = __UI_REFRESH_MS__;
    const ssRefreshMsDefault = __SS_REFRESH_MS__;
    const uiRefreshMs = Math.max(1000, Number(qs.get('refresh_ms') || uiRefreshMsDefault));
    const ssRefreshMs = Math.max(5000, Number(qs.get('statespace_refresh_ms') || ssRefreshMsDefault));
    const chartMode = (qs.get('mode') || 'brick').toLowerCase();
    const brickBaseTs = 1704067200;
    const chart = LightweightCharts.createChart(chartEl, {
      layout: { background: { color: '#1e2333' }, textColor: '#d9def7' },
      rightPriceScale: { borderColor: '#2a3044' },
      localization: {
        timeFormatter: (time) => {
          const t = Number(time);
          if (!Number.isFinite(t)) return '';
          if (chartMode !== 'brick' || !Array.isArray(barsRawRef) || !barsRawRef.length) {
            const d = new Date(t * 1000);
            return d.getUTCFullYear()+'-'+String(d.getUTCMonth()+1).padStart(2,'0')+'-'+String(d.getUTCDate()).padStart(2,'0')+' '+String(d.getUTCHours()).padStart(2,'0')+':'+String(d.getUTCMinutes()).padStart(2,'0');
          }
          const idx = Math.max(0, Math.round((t - brickBaseTs) / 60));
          if (idx >= barsRawRef.length) return 'B'+idx;
          const rt = Number(barsRawRef[idx].time);
          if (!Number.isFinite(rt)) return 'B'+idx;
          const d = new Date(rt * 1000);
          return d.getUTCFullYear()+'-'+String(d.getUTCMonth()+1).padStart(2,'0')+'-'+String(d.getUTCDate()).padStart(2,'0')+' '+String(d.getUTCHours()).padStart(2,'0')+':'+String(d.getUTCMinutes()).padStart(2,'0');
        },
      },
      timeScale: {
        borderColor: '#2a3044',
        timeVisible: chartMode !== 'brick',
        secondsVisible: false,
        tickMarkFormatter: (time) => {
          if (chartMode !== 'brick') return undefined;
          const t = Number(time);
          if (!Number.isFinite(t)) return '';
          const idx = Math.max(0, Math.round((t - brickBaseTs) / 60));
          if (!Array.isArray(barsRawRef) || idx >= barsRawRef.length) return 'B'+idx;
          const rt = Number(barsRawRef[idx].time);
          if (!Number.isFinite(rt)) return 'B'+idx;
          const d = new Date(rt * 1000);
          return String(d.getUTCDate()).padStart(2,'0')+'.'+String(d.getUTCMonth()+1).padStart(2,'0')+' '+String(d.getUTCHours()).padStart(2,'0')+':'+String(d.getUTCMinutes()).padStart(2,'0');
        },
      },
      grid: { vertLines: { color: '#252b3f' }, horzLines: { color: '#252b3f' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
    });
    const candle = chart.addCandlestickSeries({
      upColor: '#2ecc71', downColor: '#f7768e',
      borderDownColor: '#f7768e', borderUpColor: '#2ecc71',
      wickDownColor: '#f7768e', wickUpColor: '#2ecc71',
    });
    const slSeries = chart.addLineSeries({ color: '#f7768e', lineWidth: 2, title: 'SL', lastValueVisible: true, priceLineVisible: false });
    const ttpSeries = chart.addLineSeries({
      color: '#e0af68',
      lineWidth: 2,
      lineStyle: 1,
      lineType: 1,
      title: 'TTP',
      lastValueVisible: true,
      priceLineVisible: false,
    });
    const entryLineSeries = chart.addLineSeries({
      color: '#ffffff',
      lineWidth: 1,
      lineStyle: 0,
      title: 'Entry',
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    const tp1Series = chart.addLineSeries({ color: '#7aa2f7', lineWidth: 2, title: 'TP1' });
    const tp2Series = chart.addLineSeries({ color: '#bb9af7', lineWidth: 2, title: 'TP2' });
    const fibLongSeries = chart.addLineSeries({ color: '#2ecc71', lineWidth: 2, lineStyle: 0, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const fibMidSeries = chart.addLineSeries({ color: '#ffffff', lineWidth: 1, lineStyle: 2, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const fibShortSeries = chart.addLineSeries({ color: '#f7768e', lineWidth: 2, lineStyle: 0, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const priceLineSeries = chart.addLineSeries({ color: '#9aa5b1', lineWidth: 1, title: 'Last', lineStyle: 2, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const tradeSegmentSeries = [];

    let latestPayload = null;
    let timeMap = null;
    let timeAxis = [];
    let barsRawRef = [];
    let latestMid = null;
    let hasFittedOnce = false;
    const bgImages = { xy: null, xz: null, yz: null };
    let ssPayload = null;
    let lastSegmentsSig = '';
    let tickInFlight = false;
    let refreshInFlight = false;
    let pullStartY = null;
    let pullTriggered = false;

    function escapeHtml(v) {
      return String(v == null ? '' : v)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    // ── Regime gradient helpers ──
    function scoreToColor(score, alpha) {
      alpha = alpha != null ? alpha : 1.0;
      const t = (Math.max(-1, Math.min(1, score)) + 1.0) / 2.0;
      let r, g, b;
      if (t < 0.5) {
        const u = t / 0.5;
        r = 247; g = Math.round(118 + 86 * u); b = Math.round(142 * (1 - u));
      } else {
        const u = (t - 0.5) / 0.5;
        r = Math.round(247 * (1 - u) + 46 * u); g = 204; b = Math.round(113 * u);
      }
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function drawRegimeBand() {
      const canvas = document.getElementById('regime-canvas');
      if (!canvas || !latestPayload) return;
      const parent = canvas.parentElement;
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const scores = latestPayload.regime_scores || [];
      const forecast = latestPayload.regime_forecast || [];
      if (!scores.length && !forecast.length) {
        ctx.fillStyle = '#2a2e38';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        return;
      }

      const tscale = chart.timeScale();

      if (!scores.length) {
        const spans = (latestPayload.regime || {}).spans || [];
        for (const s of spans) {
          const fromT = mapTimeForChart(s.from);
          const toT = mapTimeForChart(s.to);
          const x0 = fromT != null ? tscale.timeToCoordinate(fromT) : null;
          const x1 = toT != null ? tscale.timeToCoordinate(toT) : null;
          if (x0 == null || x1 == null) continue;
          const score = Number(s.gate_on) ? 0.6 : -0.6;
          ctx.fillStyle = scoreToColor(score, 0.8);
          ctx.fillRect(Math.min(x0, x1), 0, Math.max(1, Math.abs(x1 - x0)), canvas.height);
        }
        return;
      }

      for (let i = 0; i < scores.length; i++) {
        const fromT = mapTimeForChart(scores[i].time);
        const toT = i + 1 < scores.length ? mapTimeForChart(scores[i + 1].time) : null;
        const x0 = fromT != null ? tscale.timeToCoordinate(fromT) : null;
        const x1 = toT != null ? tscale.timeToCoordinate(toT) : (x0 != null ? x0 + 2 : null);
        if (x0 == null || x1 == null) continue;
        ctx.fillStyle = scoreToColor(scores[i].score);
        ctx.fillRect(Math.min(x0, x1), 0, Math.max(1, Math.abs(x1 - x0)), canvas.height);
      }

      for (let i = 0; i < forecast.length; i++) {
        const fromT = mapTimeForChart(forecast[i].time);
        const toT = i + 1 < forecast.length ? mapTimeForChart(forecast[i + 1].time) : null;
        const x0 = fromT != null ? tscale.timeToCoordinate(fromT) : null;
        const x1 = toT != null ? tscale.timeToCoordinate(toT) : (x0 != null ? x0 + 15 : null);
        if (x0 == null || x1 == null) continue;
        const fade = 1.0 - (i / Math.max(1, forecast.length));
        ctx.fillStyle = scoreToColor(forecast[i].score, 0.3 + 0.7 * fade);
        ctx.fillRect(Math.min(x0, x1), 0, Math.max(1, Math.abs(x1 - x0)), canvas.height);
      }
    }

    // ── Heatmap helpers ──
    function loadBgImage(tag, dataUrl) {
      if (!dataUrl) return;
      const img = new Image();
      img.onload = () => { bgImages[tag] = img; drawAllHeatmaps(); };
      img.src = dataUrl;
    }

    function valToCanvas(val, size) {
      return ((val + 1.0) / 2.0) * size;
    }

    function drawHeatmap(canvasId, tag, xKey, yKey) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      const rect = canvas.parentElement.getBoundingClientRect();
      const size = Math.max(100, Math.floor(Math.min(rect.width - 24, 400)));
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      const w = size, h = size;
      ctx.clearRect(0, 0, w, h);

      if (bgImages[tag]) {
        ctx.drawImage(bgImages[tag], 0, 0, w, h);
      } else {
        ctx.fillStyle = '#181c24';
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = '#252b3f';
        ctx.lineWidth = 0.5;
        for (let v = -1; v <= 1; v += 0.25) {
          const px = valToCanvas(v, w);
          ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(0, h - px); ctx.lineTo(w, h - px); ctx.stroke();
        }
      }

      if (!ssPayload) return;

      const rd = (ssPayload.recent_density || {})[tag] || [];
      if (rd.length) {
        const maxCount = Math.max(...rd.map(c => c[2]));
        const binW = w / 28, binH = h / 28;
        for (const cell of rd) {
          const alpha = 0.1 + 0.5 * (cell[2] / Math.max(1, maxCount));
          ctx.fillStyle = `rgba(120, 180, 255, ${alpha})`;
          const cx = valToCanvas(cell[0], w) - binW / 2;
          const cy = h - valToCanvas(cell[1], h) - binH / 2;
          ctx.fillRect(cx, cy, binW, binH);
        }
      }

      const traj = ssPayload.trajectory || [];
      const cursorPt = getHeatmapCursorState();
      const sliceEnd = trajCursorIdx >= 0 ? trajCursorIdx + 1 : traj.length;
      const visibleTraj = traj.slice(0, sliceEnd);

      if (visibleTraj.length > 1) {
        for (let i = 1; i < visibleTraj.length; i++) {
          const alpha = 0.05 + 0.95 * (i / visibleTraj.length);
          ctx.strokeStyle = `rgba(100, 160, 255, ${alpha})`;
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.moveTo(valToCanvas(visibleTraj[i-1][xKey], w), h - valToCanvas(visibleTraj[i-1][yKey], h));
          ctx.lineTo(valToCanvas(visibleTraj[i][xKey], w), h - valToCanvas(visibleTraj[i][yKey], h));
          ctx.stroke();
        }
      }

      if (cursorPt) {
        const cx = valToCanvas(cursorPt[xKey], w);
        const cy = h - valToCanvas(cursorPt[yKey], h);
        ctx.strokeStyle = 'rgba(255, 80, 60, 0.55)';
        ctx.lineWidth = 0.9;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'red';
        ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = 'yellow';
        ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.fill();
      }

      ctx.fillStyle = '#8e98bf';
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      const labels = [-1, -0.5, 0, 0.5, 1];
      for (const v of labels) {
        const px = valToCanvas(v, w);
        ctx.fillText(v.toFixed(1), px, h - 2);
        ctx.fillText(v.toFixed(1), 2, h - px + 3);
      }
    }

    function drawAllHeatmaps() {
      drawHeatmap('heatmap-xy', 'xy', 'x', 'y');
      drawHeatmap('heatmap-xz', 'xz', 'x', 'z');
      drawHeatmap('heatmap-yz', 'yz', 'y', 'z');
    }

    // ── Axis status bars ──
    function drawAxisBars() {
      const canvas = document.getElementById('axis-bars');
      if (!canvas || !ssPayload) return;
      const cur = getHeatmapCursorState();
      if (!cur) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      const axes = [
        { label: 'X Drift', value: cur.x, conf: cur.conf_x, color: '#ff6644' },
        { label: 'Y Elast.', value: cur.y, conf: cur.conf_y, color: '#44bbff' },
        { label: 'Z Instab.', value: cur.z, conf: cur.conf_z, color: '#ffcc33' },
      ];

      const barH = 18, gap = 50;
      const trackLeft = 70, trackRight = w - 60;
      const trackW = trackRight - trackLeft;
      const mid = trackLeft + trackW / 2;

      axes.forEach((a, i) => {
        const y = 30 + i * gap;
        ctx.fillStyle = a.color;
        ctx.font = 'bold 11px system-ui';
        ctx.textAlign = 'left';
        ctx.fillText(a.label, 4, y + barH / 2 + 4);

        ctx.fillStyle = '#2a2e38';
        ctx.beginPath(); ctx.rect(trackLeft, y, trackW, barH); ctx.fill();

        ctx.strokeStyle = '#666'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mid, y); ctx.lineTo(mid, y + barH); ctx.stroke();

        const barEnd = mid + a.value * (trackW / 2);
        ctx.fillStyle = a.color;
        const barX = Math.min(mid, barEnd);
        ctx.beginPath(); ctx.rect(barX, y + 2, Math.abs(barEnd - mid), barH - 4); ctx.fill();

        ctx.fillStyle = '#ccc';
        ctx.font = '10px ui-monospace, monospace';
        ctx.textAlign = 'left';
        const vStr = (a.value >= 0 ? '+' : '') + a.value.toFixed(3);
        const cStr = (a.conf >= 0 ? '+' : '') + a.conf.toFixed(3);
        ctx.fillText(vStr + '  c:' + cStr, trackRight + 4, y + barH / 2 + 4);
      });
    }

    // ── Equity curve ──
        function drawEquityCurve() {
      const canvas = document.getElementById('equity-canvas');
      if (!canvas || !latestPayload) return;
      const detailEl = document.getElementById('equity-detail');
      const metaEl = document.getElementById('equity-meta');
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width);
      canvas.height = Math.floor(rect.height);
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#181c24';
      ctx.fillRect(0, 0, w, h);

      const liveEq = Array.isArray(latestPayload.equity_live) ? latestPayload.equity_live : [];
      const realizedEq = Array.isArray(latestPayload.equity_realized) ? latestPayload.equity_realized : [];
      const kucoinEq = Array.isArray(latestPayload.equity_real) ? latestPayload.equity_real : [];
      const krakenEq = Array.isArray(latestPayload.equity_kraken) ? latestPayload.equity_kraken : [];
      const combinedEq = Array.isArray(latestPayload.equity_combined) ? latestPayload.equity_combined : [];

      const liveSource = latestPayload.equity_live_source || 'none';
      const realizedSource = latestPayload.equity_realized_source || 'none';

      if (metaEl) {
        metaEl.textContent = `Equity: green=realized, orange=live (realized_source=${realizedSource}, live_source=${liveSource})`;
      }

      if (realizedEq.length >= 2 || liveEq.length >= 2) {
        const normalize = (arr) => arr.map((p) => ({ time: Number(p.time || 0), equity: Number(p.equity || 0) }))
          .filter((p) => Number.isFinite(p.time) && Number.isFinite(p.equity));

        const pRealized = normalize(realizedEq);
        const pLive = normalize(liveEq);

        const allVals = [...pRealized, ...pLive].map(p => p.equity);
        if (allVals.length >= 2) {
          const minV = Math.min(...allVals);
          const maxV = Math.max(...allVals);
          const range = Math.max(1e-6, (maxV - minV) * 1.15);
          const padL = 8, padR = 8, padT = 14, padB = 14;
          const pw = w - padL - padR, ph = h - padT - padB;

          const drawLine = (pts, color) => {
            if (!Array.isArray(pts) || pts.length < 2) return;
            ctx.lineWidth = 2;
            ctx.strokeStyle = color;
            ctx.beginPath();
            for (let i = 0; i < pts.length; i++) {
              const x = padL + (i / Math.max(1, pts.length - 1)) * pw;
              const y = padT + (1 - (pts[i].equity - minV) / range) * ph;
              if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
          };

          drawLine(pRealized, '#9ece6a');
          drawLine(pLive, '#ff9e64');

          const anchor = pRealized.length ? pRealized : pLive;
          const delta = anchor.length >= 2 ? (anchor[anchor.length - 1].equity - anchor[0].equity) : 0;
          const pct = (anchor.length >= 1 && anchor[0].equity > 0) ? (delta / anchor[0].equity) * 100.0 : 0.0;

          ctx.font = 'bold 11px system-ui';
          ctx.textAlign = 'right';
          ctx.fillStyle = delta >= 0 ? '#9ece6a' : '#f7768e';
          ctx.fillText(`${delta >= 0 ? '+' : ''}${delta.toFixed(2)} USDT (${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%)`, w - padR, padT - 2);

          const latestRealized = pRealized.length ? pRealized[pRealized.length - 1].equity : null;
          const latestLive = pLive.length ? pLive[pLive.length - 1].equity : null;

          if (detailEl) {
            detailEl.textContent = `realized:${latestRealized !== null ? latestRealized.toFixed(2) : '-'} live:${latestLive !== null ? latestLive.toFixed(2) : '-'}`;
          }
          canvas.onmousemove = null;
          return;
        }
      }

      const trades = Array.isArray(latestPayload.diary_entries) && latestPayload.diary_entries.length
        ? latestPayload.diary_entries
        : (latestPayload.equity_curve || []);
      if (!trades.length) {
        ctx.fillStyle = '#8e98bf';
        ctx.font = '11px system-ui';
        ctx.textAlign = 'center';
        const op = latestPayload.open_position;
        if (op && op.side && op.entry_time) {
          const d = new Date(Number(op.entry_time) * 1000);
          const hh = String(d.getUTCHours()).padStart(2, '0');
          const mm = String(d.getUTCMinutes()).padStart(2, '0');
          ctx.fillText('No closed trades (open position)', w / 2, h / 2 - 6);
          const ep = Number.isFinite(Number(op.entry_price)) ? Number(op.entry_price).toFixed(3) : '-';
          const sl = Number.isFinite(Number(op.sl)) ? Number(op.sl).toFixed(3) : '-';
          ctx.fillText(`${hh}:${mm} ${String(op.side).toUpperCase()} entry:${ep} SL:${sl}`, w / 2, h / 2 + 12);
          if (detailEl) detailEl.textContent = `open ${String(op.side).toUpperCase()} entry:${ep} SL:${sl}`;
        } else {
          ctx.fillText('No closed trades', w / 2, h / 2);
          if (detailEl) detailEl.textContent = '-';
        }
        canvas.onmousemove = null;
        return;
      }

      let running = 0.0;
      const points = trades.map((t) => {
        const p = Number(t.pnl_pct || 0);
        running += p;
        return {
          time: Number(t.time || 0),
          pnl_pct: p,
          cum_pct: Number.isFinite(Number(t.cum_pct)) ? Number(t.cum_pct) : running,
          side: t.side || '-',
          entry_price: t.entry_price,
          exit_price: t.exit_price,
          qty: t.qty,
          source: t.source || realizedSource || latestPayload.diary_source || latestPayload.equity_source || 'none',
        };
      });

      const vals = points.map(t => t.cum_pct);
      const minV = Math.min(0, ...vals);
      const maxV = Math.max(0, ...vals);
      const range = Math.max(1, maxV - minV) * 1.15;
      const padL = 8, padR = 8, padT = 14, padB = 14;
      const pw = w - padL - padR, ph = h - padT - padB;

      function tx(i) { return padL + (i / Math.max(1, points.length - 1)) * pw; }
      function ty(v) { return padT + (1 - (v - minV) / range) * ph; }

      const y0 = ty(0);
      ctx.strokeStyle = '#3a3f52';
      ctx.lineWidth = 0.7;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(padL, y0); ctx.lineTo(w - padR, y0); ctx.stroke();
      ctx.setLineDash([]);

      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < points.length; i++) {
        const x = tx(i);
        const y = ty(points[i].cum_pct);
        if (i === 0) { ctx.moveTo(x, y0); ctx.lineTo(x, y); }
        else {
          ctx.lineTo(x, ty(points[i - 1].cum_pct));
          ctx.lineTo(x, y);
        }
      }
      const lastCum = points[points.length - 1].cum_pct;
      ctx.strokeStyle = lastCum >= 0 ? '#9ece6a' : '#f7768e';
      ctx.stroke();

      ctx.font = 'bold 9px system-ui';
      ctx.textAlign = 'center';
      for (let i = 0; i < points.length; i++) {
        const x = tx(i);
        const y = ty(points[i].cum_pct);
        const pnl = points[i].pnl_pct;
        ctx.fillStyle = pnl >= 0 ? '#9ece6a' : '#f7768e';
        const label = (pnl >= 0 ? '+' : '') + pnl.toFixed(1) + '%';
        ctx.fillText(label, x, y - 5);
      }

      ctx.font = 'bold 11px system-ui';
      ctx.textAlign = 'right';
      ctx.fillStyle = lastCum >= 0 ? '#9ece6a' : '#f7768e';
      ctx.fillText((lastCum >= 0 ? '+' : '') + lastCum.toFixed(1) + '%', w - padR, padT - 2);

      canvas.onmousemove = (ev) => {
        if (!detailEl) return;
        const r = canvas.getBoundingClientRect();
        const x = ev.clientX - r.left;
        let idx = Math.round(((x - padL) / Math.max(1, pw)) * Math.max(1, points.length - 1));
        idx = Math.max(0, Math.min(points.length - 1, idx));
        const t = points[idx];
        const d = new Date(Number(t.time) * 1000);
        const hh = String(d.getUTCHours()).padStart(2, '0');
        const mm = String(d.getUTCMinutes()).padStart(2, '0');
        const ep = Number.isFinite(Number(t.entry_price)) ? Number(t.entry_price).toFixed(3) : '-';
        const xp = Number.isFinite(Number(t.exit_price)) ? Number(t.exit_price).toFixed(3) : '-';
        const q = Number.isFinite(Number(t.qty)) ? Number(t.qty).toFixed(2) : '-';
        detailEl.textContent = `${hh}:${mm} ${String(t.side || '-').toUpperCase()} q:${q} ${ep}→${xp} pnl:${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}% cum:${t.cum_pct >= 0 ? '+' : ''}${t.cum_pct.toFixed(2)}%`;
      };
      if (detailEl) {
        const t = points[points.length - 1];
        detailEl.textContent = `latest ${String(t.side || '-').toUpperCase()} pnl:${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}% cum:${t.cum_pct >= 0 ? '+' : ''}${t.cum_pct.toFixed(2)}%`;
      }
    }

    // ── Time cursor for heatmaps ──
    let trajCursorIdx = -1;  // -1 = live/latest

    function getHeatmapCursorState() {
      if (!ssPayload || !ssPayload.trajectory || !ssPayload.trajectory.length) return null;
      const traj = ssPayload.trajectory;
      if (trajCursorIdx < 0 || trajCursorIdx >= traj.length) return ssPayload.current;
      const pt = traj[trajCursorIdx];
      return { x: pt.x, y: pt.y, z: pt.z, conf_x: 0, conf_y: 0, conf_z: 0 };
    }

    // ── Time mapping (unchanged) ──
    function buildTimeMapFromBars(bars) {
      const m = new Map();
      const arr = [];
      if (!Array.isArray(bars)) return m;
      for (let i = 0; i < bars.length; i++) {
        const t = Number(bars[i].time);
        if (!Number.isFinite(t)) continue;
        arr.push(t);
        if (!m.has(t)) m.set(t, i);
      }
      timeAxis = arr;
      return m;
    }

    function mapTimeForChart(t) {
      if (t == null) return null;
      const n = Number(t);
      if (!Number.isFinite(n)) return null;
      if (chartMode !== 'brick' || !timeMap) return n;
      let idx = timeMap.get(n);
      if (idx == null && Array.isArray(barsRawRef) && barsRawRef.length) {
        let lo = 0, hi = barsRawRef.length - 1, best = null;
        while (lo <= hi) {
          const m = Math.floor((lo + hi) / 2);
          const tv = Number(barsRawRef[m].time);
          if (tv <= n) { best = m; lo = m + 1; } else { hi = m - 1; }
        }
        idx = best;
      }
      if (idx == null) return null;
      return brickBaseTs + idx * 60;
    }

    function mapTimeAsOfForChart(t) {
      if (t == null) return null;
      const n = Number(t);
      if (!Number.isFinite(n)) return null;
      if (chartMode !== 'brick' || !Array.isArray(timeAxis) || timeAxis.length === 0) return n;
      let lo = 0, hi = timeAxis.length - 1, ans = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (timeAxis[mid] <= n) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
      }
      if (ans < 0) return null;
      return brickBaseTs + ans * 60;
    }

    function mapBarsForChart(bars) {
      if (!Array.isArray(bars)) return [];
      if (chartMode !== 'brick') return bars;
      return bars.map((b, i) => ({ ...b, time: brickBaseTs + i * 60 }));
    }

    function mapMarkersForChart(markers) {
      if (!Array.isArray(markers)) return [];
      if (chartMode !== 'brick') return markers;
      return markers.map((m) => {
        const mapped = mapTimeAsOfForChart(m.time);
        if (mapped == null) return null;
        return { ...m, time: mapped };
      }).filter(Boolean);
    }

    function mapLineForChart(points) {
      if (!Array.isArray(points)) return [];
      if (chartMode !== 'brick') return points;
      return points.map((p) => {
        const mapped = mapTimeForChart(p.time);
        if (mapped == null) return null;
        return { time: mapped, value: Number(p.value) };
      }).filter(Boolean);
    }

    function mapSegmentForChart(seg) {
      const t0 = mapTimeForChart(seg.from_time);
      const t1 = mapTimeForChart(seg.to_time);
      if (t0 == null || t1 == null) return [];
      return [{ time: t0, value: Number(seg.from_price) }, { time: t1, value: Number(seg.to_price) }];
    }

    function buildUnifiedExitLine(bars, levels) {
      if (!Array.isArray(bars) || !bars.length || !levels) return { data: [], mode: 'none' };
      const sl = Number(levels.sl);
      const ttp = Number(levels.ttp);
      const hasSl = Number.isFinite(sl);
      const hasTtp = Number.isFinite(ttp);
      if (!hasSl && !hasTtp) return { data: [], mode: 'none' };
      let entryTime = null;
      if (levels.entry_bar_ts != null) {
        entryTime = mapTimeForChart(Number(levels.entry_bar_ts));
      }
      if (entryTime == null) {
        if (levels.side) {
          const startIdx = Math.max(0, bars.length - Math.round(bars.length * 0.2));
          entryTime = bars[startIdx].time;
        } else {
          entryTime = bars[0].time;
        }
      }
      const lastTime = bars[bars.length - 1].time;
      if (hasTtp) {
        const side = String(levels.side || '').toLowerCase();
        let exitVal = ttp;
        if (hasSl) {
          exitVal = (side === 'short') ? Math.min(sl, ttp) : Math.max(sl, ttp);
        }
        return { data: [{ time: entryTime, value: exitVal }, { time: lastTime, value: exitVal }], mode: 'ttp' };
      }
      return { data: [{ time: entryTime, value: sl }, { time: lastTime, value: sl }], mode: 'sl' };
    }

    function buildTTPTrail(bars, levels, ttpTrailPct) {
      if (!Array.isArray(bars) || !bars.length || !levels) return [];
      const entryPx = Number(levels.entry_px);
      const sideStr = String(levels.side || '').toLowerCase();
      if (!Number.isFinite(entryPx) || !sideStr) return [];
      const isLong = sideStr === 'long' || sideStr === 'l' || sideStr === '1';
      const trail = Number.isFinite(Number(ttpTrailPct)) && Number(ttpTrailPct) > 0 ? Number(ttpTrailPct) : 0.012;
      let entryT = null;
      if (levels.entry_bar_ts != null) {
        entryT = mapTimeForChart(Number(levels.entry_bar_ts));
      }
      if (entryT == null) return [];
      let startIdx = bars.length - 1;
      for (let i = 0; i < bars.length; i++) {
        if (Number(bars[i].time) >= Number(entryT)) { startIdx = i; break; }
      }
      let bestFav = entryPx;
      const points = [];
      for (let i = startIdx; i < bars.length; i++) {
        const h = Number(bars[i].high || bars[i].close);
        const l = Number(bars[i].low || bars[i].close);
        if (isLong) {
          bestFav = Math.max(bestFav, h);
          points.push({ time: bars[i].time, value: bestFav * (1 - trail) });
        } else {
          bestFav = Math.min(bestFav, l);
          points.push({ time: bars[i].time, value: bestFav * (1 + trail) });
        }
      }
      return points;
    }

    function fmtNum(v) {
      if (v == null || Number.isNaN(Number(v))) return '-';
      return Number(v).toFixed(4);
    }

    function levelLineData(lastBars, level) {
      if (!Array.isArray(lastBars) || !lastBars.length || level == null || Number.isNaN(Number(level))) return [];
      const first = lastBars[0].time;
      const last = lastBars[lastBars.length - 1].time;
      return [{ time: first, value: Number(level) }, { time: last, value: Number(level) }];
    }

    function levelLineFromEntry(lastBars, level, levels) {
      if (!Array.isArray(lastBars) || !lastBars.length || level == null || Number.isNaN(Number(level))) return [];
      let first = lastBars[0].time;
      if (levels && levels.entry_bar_ts != null) {
        const mapped = mapTimeForChart(Number(levels.entry_bar_ts));
        if (mapped != null) first = mapped;
      }
      const last = lastBars[lastBars.length - 1].time;
      return [{ time: first, value: Number(level) }, { time: last, value: Number(level) }];
    }

    function liveRegimeScore(payload) {
      const gc = (payload && payload.gate_confidence) ? payload.gate_confidence : null;
      if (!gc) return null;
      const pTrend = Number(gc.selected_p_trend);
      if (!Number.isFinite(pTrend)) return null;
      return Math.max(-1, Math.min(1, 2 * pTrend - 1));
    }

    function getChartRangeParams() {
      const v = (document.getElementById('chart-range') || {}).value || '14d';
      if (v === '30d') return { hours: 24 * 30, max_points: 9000 };
      if (v === 'all') return { hours: 24 * 120, max_points: 20000 };
      return { hours: 24 * 14, max_points: 4000 };
    }

    // ── Data loading ──
    async function loadMeta() {
      const [st, pos] = await Promise.all([
        fetch('/api/status').then(r => r.json()),
        fetch('/api/position').then(r => r.json())
      ]);
      document.getElementById('api-status').textContent = st.api_configured ? 'configured' : 'missing';
      document.getElementById('api-status').className = st.api_configured ? 'ok' : 'err';
      if (st.hint) document.getElementById('hint').textContent = st.hint;
      if (st.ticker) {
        const t = st.ticker;
        const b = typeof t.bid === 'number' ? t.bid.toFixed(4) : t.bid;
        const a = typeof t.ask === 'number' ? t.ask.toFixed(4) : t.ask;
        const m = t.mid != null ? (typeof t.mid === 'number' ? t.mid.toFixed(4) : t.mid) : null;
        document.getElementById('ticker').textContent = m != null ? `${m} (bid ${b} / ask ${a})` : `${b} / ${a}`;
        latestMid = Number(t.mid);
      } else {
        document.getElementById('ticker').textContent = st.ticker_error || '-';
        latestMid = null;
      }
      if (pos.position != null) {
        const lev = (pos.leverage != null && Number.isFinite(Number(pos.leverage))) ? (' x' + Number(pos.leverage).toFixed(1)) : '';
        document.getElementById('position').textContent = String(pos.position) + lev;
      } else {
        document.getElementById('position').textContent = pos.error || '-';
      }
      if (pos.position != null && st.ticker && st.ticker.mid != null) {
        const mult = Number(pos.contract_multiplier || 1);
        const notional = Math.abs(Number(pos.position)) * mult * Number(st.ticker.mid);
        document.getElementById('position-notional').textContent = Number.isFinite(notional) ? notional.toFixed(2)+' USDT' : '-';
      } else {
        document.getElementById('position-notional').textContent = '-';
      }
      const bal = st.balance;
      if (bal && bal.equity != null) {
        document.getElementById('capital').textContent = Number(bal.equity).toFixed(2) + ' USDT';
      } else {
        document.getElementById('capital').textContent = '-';
      }
    }

    async function loadChart() {
      const p = getChartRangeParams();
      const chartUrl = '/api/dashboard/chart?hours=' + p.hours + '&max_points=' + p.max_points;
      const payload = await fetch(chartUrl).then(r => r.json());
      latestPayload = payload;
      if (!payload.ok) return;
      const prevLogicalRange = hasFittedOnce ? chart.timeScale().getVisibleLogicalRange() : null;

      const barsRaw = Array.isArray(payload.bars) ? payload.bars : [];
      timeMap = buildTimeMapFromBars(barsRaw);
      const bars = mapBarsForChart(barsRaw);
      candle.setData(bars);
      candle.setMarkers(mapMarkersForChart(Array.isArray(payload.markers) ? payload.markers : []));

      barsRawRef = barsRaw;

      const levels = payload.levels || {};
      const exitInfo = buildUnifiedExitLine(bars, levels);
      slSeries.setData(levelLineFromEntry(bars, levels.sl, levels));

      const ttpTrailData = buildTTPTrail(bars, levels, payload.ttp_trail_pct);
      ttpSeries.setData(ttpTrailData.length > 0 ? ttpTrailData : levelLineFromEntry(bars, levels.ttp, levels));
      entryLineSeries.setData(levelLineFromEntry(bars, levels.entry_px, levels));

      tp1Series.setData(levelLineData(bars, levels.tp1));
      tp2Series.setData(levelLineData(bars, levels.tp2));

      const fibo = payload.fibo || {};
      fibLongSeries.setData(mapLineForChart(fibo.long || []));
      fibMidSeries.setData(mapLineForChart(fibo.mid || []));
      fibShortSeries.setData(mapLineForChart(fibo.short || []));

      const lastBar = bars.length ? bars[bars.length - 1] : null;
      if (lastBar) {
        const livePx = Number.isFinite(Number(latestMid)) ? Number(latestMid) : Number(lastBar.close);
        priceLineSeries.setData([{ time: bars[0].time, value: livePx }, { time: lastBar.time, value: livePx }]);
      } else {
        priceLineSeries.setData([]);
      }

      const segments = Array.isArray(payload.segments) ? payload.segments : [];
      const segSig = JSON.stringify(segments.map((s) => [s.from_time, s.to_time, s.from_price, s.to_price, s.color, !!s.positive]));
      if (segSig !== lastSegmentsSig) {
        for (const s of tradeSegmentSeries) chart.removeSeries(s);
        tradeSegmentSeries.length = 0;
        for (const seg of segments) {
          const ls = chart.addLineSeries({ color: seg.color || '#9aa5b1', lineWidth: 2, title: seg.positive ? 'Trade +' : 'Trade -' });
          ls.setData(mapSegmentForChart(seg));
          tradeSegmentSeries.push(ls);
        }
        lastSegmentsSig = segSig;
      }

      const exitModeEl = document.getElementById('exit-mode');
      if (exitInfo.mode === 'ttp') {
        exitModeEl.textContent = 'TTP (trailing)';
        exitModeEl.className = 'ok';
      } else if (exitInfo.mode === 'sl') {
        exitModeEl.textContent = 'SL (stop loss)';
        exitModeEl.className = 'err';
      } else {
        exitModeEl.textContent = '-';
        exitModeEl.className = '';
      }

      document.getElementById('lvl-sl').textContent = fmtNum(levels.sl);
      document.getElementById('lvl-ttp').textContent = fmtNum(levels.ttp);
      document.getElementById('lvl-tp1').textContent = fmtNum(levels.tp1);
      document.getElementById('lvl-tp2').textContent = fmtNum(levels.tp2);
      const barTimeEl = document.getElementById('bar-time');
      let barTs = null;
      if (levels && levels.entry_bar_ts != null && Number.isFinite(Number(levels.entry_bar_ts))) {
        barTs = Number(levels.entry_bar_ts);
      } else if (payload.open_position && payload.open_position.entry_time != null && Number.isFinite(Number(payload.open_position.entry_time))) {
        barTs = Number(payload.open_position.entry_time);
      }
      if (barTimeEl) {
        if (barTs != null) {
          const d = new Date(barTs * 1000);
          const yy = d.getUTCFullYear();
          const mo = String(d.getUTCMonth() + 1).padStart(2, '0');
          const dd = String(d.getUTCDate()).padStart(2, '0');
          const hh = String(d.getUTCHours()).padStart(2, '0');
          const mm = String(d.getUTCMinutes()).padStart(2, '0');
          barTimeEl.textContent = `${yy}-${mo}-${dd} ${hh}:${mm} UTC`;
        } else {
          barTimeEl.textContent = '-';
        }
      }

      document.getElementById('regime-state').textContent = payload.regime_state || '-';
      const conf = payload.confidence == null ? null : Number(payload.confidence);
      document.getElementById('confidence').textContent = conf == null ? '-' : conf.toFixed(3);
      if (conf != null) {
        const score = liveRegimeScore(payload);
        if (Number.isFinite(score)) {
          document.getElementById('confidence').style.color = score >= 0 ? '#9ece6a' : '#f7768e';
        } else {
          const rs = String(payload.regime_state || '').toLowerCase();
          if (rs === 'trend') {
            document.getElementById('confidence').style.color = conf >= 0.7 ? '#9ece6a' : (conf >= 0.5 ? '#e0af68' : '#f7768e');
          } else {
            document.getElementById('confidence').style.color = conf >= 0.7 ? '#f7768e' : (conf >= 0.5 ? '#e0af68' : '#9ece6a');
          }
        }
      }

      if (!hasFittedOnce) {
        chart.timeScale().fitContent();
        hasFittedOnce = true;
      } else if (prevLogicalRange) {
        chart.timeScale().setVisibleLogicalRange(prevLogicalRange);
      }
      drawRegimeBand();
      drawEquityCurve();
    }

    async function loadStateSpace() {
      const windowH = document.getElementById('traj-window').value || '8';
      try {
        const data = await fetch('/api/dashboard/statespace?window_hours=' + windowH).then(r => r.json());
        if (!data.ok) return;
        ssPayload = data;
        const bg = data.density_bg || {};
        for (const tag of ['xy', 'xz', 'yz']) {
          if (bg[tag] && !bgImages[tag]) loadBgImage(tag, bg[tag]);
        }
        const slider = document.getElementById('traj-slider');
        if (Number(slider.value) >= 100) {
          trajCursorIdx = -1;
          document.getElementById('traj-slider-label').textContent = 'now';
        }
        drawAllHeatmaps();
        drawAxisBars();
      } catch (e) { /* state space unavailable */ }
    }

    async function loadFills() {
      const host = document.getElementById('fills-list');
      if (!host) return;
      try {
        const data = await fetch('/api/dashboard/fills?max_points=200').then(r => r.json());
        if (!data.ok) return;
        const rows = Array.isArray(data.rows) ? data.rows : [];
        const lines = ['<div class=\"fills-row head\"><span>time (UTC)</span><span>side</span><span>qty</span><span>price</span><span>reason</span></div>'];
        for (let i = Math.max(0, rows.length - 80); i < rows.length; i++) {
          const r = rows[i] || {};
          const ts = (typeof r.time_utc === 'string' && r.time_utc) ? r.time_utc : '-';
          const side = String(r.side || '-').toLowerCase();
          const cls = side === 'buy' ? 'fills-buy' : (side === 'sell' ? 'fills-sell' : '');
          const qty = Number.isFinite(Number(r.size)) ? Number(r.size).toFixed(2) : '-';
          const px = Number.isFinite(Number(r.price)) ? Number(r.price).toFixed(3) : '-';
          const reasonBase = String(r.reason || '-');
          const cid = (typeof r.client_oid === 'string' && r.client_oid) ? ` [${r.client_oid}]` : '';
          const reason = reasonBase + cid;
          lines.push(
            `<div class=\"fills-row\"><span>${escapeHtml(ts)}</span><span class=\"${cls}\">${escapeHtml(side)}</span><span>${escapeHtml(qty)}</span><span>${escapeHtml(px)}</span><span>${escapeHtml(reason)}</span></div>`
          );
        }
        host.innerHTML = lines.join('');
      } catch (e) { /* fills unavailable */ }
    }

    async function refreshNow(reason) {
      if (refreshInFlight) return;
      refreshInFlight = true;
      const btn = document.getElementById('chart-refresh-btn');
      if (btn) {
        btn.disabled = true;
        btn.textContent = 'Refreshing...';
      }
      try {
        await Promise.all([tick(), loadStateSpace()]);
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.textContent = 'Refresh';
        }
        refreshInFlight = false;
      }
    }

    async function sendManualOrder(actionOverride) {
      const actionEl = document.getElementById('manual-action');
      const qtyEl = document.getElementById('manual-qty');
      const resultEl = document.getElementById('manual-result');
      if (!actionEl || !qtyEl || !resultEl) return;
      const action = String(actionOverride || actionEl.value || '').trim();
      if (!action) return;
      const qtyRaw = String(qtyEl.value || '').trim();
      const payload = { action };
      const isEntry = action === 'enter_long' || action === 'enter_short';
      if (qtyRaw.length) {
        const q = Number(qtyRaw);
        if (Number.isFinite(q) && q > 0) payload.qty = q;
      }
      if (isEntry && payload.qty == null) {
        resultEl.textContent = 'qty is required for enter_long / enter_short';
        return;
      }
      const headers = { 'Content-Type': 'application/json' };
      const tok = String(qs.get('token') || '').trim();
      if (tok) headers['x-webhook-token'] = tok;
      resultEl.textContent = 'sending...';
      try {
        const data = await fetch('/api/manual/order', {
          method: 'POST',
          headers,
          body: JSON.stringify(payload),
        }).then(r => r.json());
        resultEl.textContent = JSON.stringify(data);
        await Promise.all([loadMeta(), loadFills(), loadChart()]);
      } catch (e) {
        resultEl.textContent = String(e && e.message ? e.message : e);
      }
    }

    chart.timeScale().subscribeVisibleTimeRangeChange(() => drawRegimeBand());
    window.addEventListener('resize', () => { drawRegimeBand(); drawAllHeatmaps(); drawEquityCurve(); });
    document.getElementById('traj-window').addEventListener('change', loadStateSpace);
    document.getElementById('chart-range').addEventListener('change', () => {
      hasFittedOnce = false;
      loadChart();
    });

    document.getElementById('traj-slider').addEventListener('input', function() {
      const slider = this;
      const pct = Number(slider.value);
      const label = document.getElementById('traj-slider-label');
      if (!ssPayload || !ssPayload.trajectory || !ssPayload.trajectory.length) {
        trajCursorIdx = -1;
        label.textContent = 'now';
        return;
      }
      const traj = ssPayload.trajectory;
      if (pct >= 100) {
        trajCursorIdx = -1;
        label.textContent = 'now';
      } else {
        trajCursorIdx = Math.round(pct / 100 * (traj.length - 1));
        const pt = traj[trajCursorIdx];
        if (pt && pt.ts) {
          const d = new Date(pt.ts * 1000);
          label.textContent = String(d.getUTCHours()).padStart(2,'0')+':'+String(d.getUTCMinutes()).padStart(2,'0');
        } else {
          label.textContent = '#' + trajCursorIdx;
        }
      }
      drawAllHeatmaps();
      drawAxisBars();
    });

    async function tick() {
      if (tickInFlight) return;
      tickInFlight = true;
      try {
        await Promise.all([loadMeta(), loadChart(), loadFills()]);
      } finally {
        tickInFlight = false;
      }
    }
    tick();
    loadStateSpace();
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') refreshNow('visible');
    });
    window.addEventListener('focus', () => refreshNow('focus'));
    window.addEventListener('pageshow', () => refreshNow('pageshow'));
    window.addEventListener('touchstart', (ev) => {
      if (!ev.touches || ev.touches.length !== 1) return;
      const topY = window.scrollY || document.documentElement.scrollTop || 0;
      if (topY <= 2) {
        pullStartY = ev.touches[0].clientY;
        pullTriggered = false;
      }
    }, { passive: true });
    window.addEventListener('touchmove', (ev) => {
      if (pullStartY == null || pullTriggered) return;
      if (!ev.touches || ev.touches.length !== 1) return;
      const dy = ev.touches[0].clientY - pullStartY;
      if (dy >= 90) {
        pullTriggered = true;
        refreshNow('pull');
      }
    }, { passive: true });
    window.addEventListener('touchend', () => {
      pullStartY = null;
      pullTriggered = false;
    }, { passive: true });
    document.getElementById('manual-send').addEventListener('click', () => sendManualOrder(null));
    document.getElementById('manual-cancel-short').addEventListener('click', () => sendManualOrder('cancel_short'));
    document.getElementById('manual-cancel-long').addEventListener('click', () => sendManualOrder('cancel_long'));
    setInterval(tick, uiRefreshMs);
    setInterval(loadStateSpace, ssRefreshMs);
  </script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """Simple dashboard UI: API status, SOL ticker from KuCoin, position. Basis für spätere Desktop-App."""
    try:
        ui_ms = int(float(os.getenv("DASHBOARD_UI_REFRESH_MS", "4000")))
    except Exception:
        ui_ms = 4000
    try:
        ss_ms = int(float(os.getenv("DASHBOARD_STATESPACE_REFRESH_MS", "15000")))
    except Exception:
        ss_ms = 15000
    ui_ms = max(1000, ui_ms)
    ss_ms = max(5000, ss_ms)
    return (
        DASHBOARD_HTML
        .replace("__UI_REFRESH_MS__", str(ui_ms))
        .replace("__SS_REFRESH_MS__", str(ss_ms))
    )


@app.post("/webhook/tradingview")
async def tradingview_webhook(
    request: Request,
    x_webhook_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    if _auth_required():
        _check_token(x_webhook_token)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")

    sym = _normalize_symbol(_symbol_from_payload(payload))
    day = _today_utc()

    out_dir = _signals_root() / sym
    _ensure_dir(out_dir)
    out_path = out_dir / f"{day}.jsonl"

    now_iso = _now_utc_iso()
    enriched = {
        "server_ts": now_iso,
        **_ensure_ts(payload, now_iso),
    }

    _append_jsonl(out_path, enriched)
    log.info(f"webhook saved symbol={sym} file={out_path}")
    return {"ok": True, "saved_to": str(out_path), "symbol": sym}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TradingView webhook + dashboard (24/7: use --host 0.0.0.0)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (0.0.0.0 for cloud)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    return p.parse_args()


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    args = parse_args()
    _sync_gate_conf_artifacts_if_enabled()
    _start_renko_cache_updater_if_enabled()
    _start_live_signal_worker_if_enabled()
    _start_live_executor_if_enabled()
    # Railway/cloud set PORT; use it so the app listens on the right port
    port = int(os.environ.get("PORT", str(args.port)))
    uvicorn.run(
        "quant.execution.webhook_server:app",
        host=args.host,
        port=port,
        reload=bool(args.reload),
        log_level="info",
    )
