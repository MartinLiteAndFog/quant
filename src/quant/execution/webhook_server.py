# src/quant/execution/webhook_server.py

from __future__ import annotations

import argparse
import json
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
    build_equity_curve,
    build_fibo_levels,
    build_regime_overlay,
    build_regime_scores,
    load_active_levels,
    load_live_fill_markers,
    load_renko_bars,
    load_renko_health,
    load_trade_segments,
    load_trade_markers,
)
from quant.execution.dashboard_statespace import (
    load_state_space_trajectory,
    compute_recent_density,
    refresh_state_space_cache,
)
from quant.regime import RegimeStore, get_live_gate_confidence
from ..utils.log import get_logger

log = get_logger("quant.webhook")


def _state_space_refresh_loop() -> None:
    interval = int(os.getenv("DASHBOARD_SS_REFRESH_SEC", "300"))
    while True:
        try:
            info = refresh_state_space_cache()
            if info.get("ok"):
                log.info("state space refresh: %d rows", info.get("rows", 0))
        except Exception as e:
            log.warning("state space refresh failed: %s", e)
        time.sleep(max(60, interval))


@asynccontextmanager
async def _lifespan(a: FastAPI):
    t = threading.Thread(target=_state_space_refresh_loop, daemon=True, name="ss-refresh")
    t.start()
    log.info("state space refresh thread started (interval=%ss)", os.getenv("DASHBOARD_SS_REFRESH_SEC", "300"))
    yield


app = FastAPI(title="quant-webhook", version="0.1.0", lifespan=_lifespan)

# Default symbol for dashboard ticker/position
DEFAULT_SYMBOL = os.getenv("DASHBOARD_SYMBOL", "SOL-USDT")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _today_utc() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d")


def _now_utc_iso() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


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


def _auth_required() -> bool:
    return bool(os.getenv("WEBHOOK_TOKEN", "").strip())


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _start_renko_cache_updater_if_enabled() -> None:
    """
    Optional background updater for dashboard Renko cache.
    Controlled via env:
      ENABLE_DASHBOARD_RENKO_UPDATER=1
    """
    if not _truthy(os.getenv("ENABLE_DASHBOARD_RENKO_UPDATER", "0")):
        return

    symbol = os.getenv("DASHBOARD_SYMBOL", "SOL-USDT")
    out_parquet = os.getenv("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet")
    box = float(os.getenv("DASHBOARD_RENKO_BOX", "0.1"))
    days_back = int(os.getenv("DASHBOARD_RENKO_DAYS_BACK", "14"))
    step_hours = int(os.getenv("DASHBOARD_RENKO_STEP_HOURS", "6"))
    poll_sec = float(os.getenv("DASHBOARD_RENKO_POLL_SEC", "300"))

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
                log.info("dashboard renko updater: %s", info)
            except Exception as e:
                log.warning("dashboard renko updater failed: %s", e)
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
    key = (os.getenv("KUCOIN_FUTURES_API_KEY") or "").strip()
    out = {"ok": True, "ts": _now_utc_iso(), "api_configured": bool(key)}
    if not key:
        out["ticker"] = None
        out["balance"] = None
        out["hint"] = "Set KUCOIN_FUTURES_API_KEY, KUCOIN_FUTURES_API_SECRET, KUCOIN_FUTURES_PASSPHRASE (e.g. in .env or cloud env vars)."
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
    return out


@app.get("/api/position")
def api_position(symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """Current position from KuCoin Futures (signed: >0 long, <0 short)."""
    key = (os.getenv("KUCOIN_FUTURES_API_KEY") or "").strip()
    if not key:
        return {"ok": True, "symbol": symbol, "position": None, "hint": "Configure KuCoin API keys."}
    try:
        broker = _kucoin_broker()
        pos = broker.get_position(symbol)
        return {"ok": True, "symbol": symbol, "position": pos}
    except Exception as e:
        return {"ok": False, "symbol": symbol, "position": None, "error": str(e)}


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
def api_dashboard_chart(symbol: str = DEFAULT_SYMBOL, hours: int = 24 * 7, max_points: int = 3000) -> Dict[str, Any]:
    """
    Unified chart payload: renko bars, trades, regime overlays, and active levels.
    """
    try:
        bars = load_renko_bars(max_points=int(max(100, max_points)))
        markers = load_trade_markers(max_points=int(max(1000, max_points * 50)))
        markers_live = load_live_fill_markers(symbol=symbol, limit=int(max(50, min(500, max_points))))
        markers = sorted(markers + markers_live, key=lambda x: int(x.get("time", 0)))
        if len(markers) > int(max(100, max_points)):
            markers = markers[-int(max(100, max_points)):]
        segments = load_trade_segments(max_points=int(max(100, max_points)))
        levels = load_active_levels()
        fibo = build_fibo_levels(max_points=int(max(100, max_points)))
        renko_health = load_renko_health()
        regime = build_regime_overlay(symbol=symbol, hours=int(max(1, hours)))
        latest = regime.get("latest") or {}
        live_gc = None
        live_gc_error = None
        try:
            live_gc = get_live_gate_confidence()
        except Exception as e:
            live_gc_error = str(e)
            log.warning("live gate confidence unavailable: %s", e)
        selected_p_trend = (live_gc or {}).get("selected_p_trend")
        live_conf = float(max(0.0, min(1.0, selected_p_trend))) if isinstance(selected_p_trend, (float, int)) else None
        if live_conf is not None and isinstance(regime.get("latest"), dict):
            regime["latest"]["confidence"] = live_conf
        if live_conf is not None and isinstance(regime.get("spans"), list) and regime["spans"]:
            regime["spans"][-1]["confidence"] = live_conf
        confidence_out = live_conf if live_conf is not None else latest.get("confidence")

        regime_score_data = build_regime_scores(symbol=symbol, hours=int(max(1, hours)))
        equity = build_equity_curve(max_points=int(max(100, max_points)))

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

        return {
            "ok": True,
            "symbol": symbol,
            "bars": bars,
            "markers": markers,
            "levels": levels,
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
            "error": str(e),
            "ts": _now_utc_iso(),
        }


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
      <p id="hint" class="hint"></p>
    </div>
  </div>

  <div class="card regime-band" style="margin-top:0.5rem;grid-column:1;">
    <canvas id="regime-canvas"></canvas>
  </div>
  <div class="hint" style="text-align:center;margin-top:0.25rem;">Regime: red = countertrend, green = trend. Right side = projected.</div>

  <div class="traj-controls">
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
    const ttpSeries = chart.addLineSeries({ color: '#ffcc66', lineWidth: 2, title: 'TTP', lastValueVisible: true, priceLineVisible: false });
    const tp1Series = chart.addLineSeries({ color: '#7aa2f7', lineWidth: 2, title: 'TP1' });
    const tp2Series = chart.addLineSeries({ color: '#bb9af7', lineWidth: 2, title: 'TP2' });
    const fibLongSeries = chart.addLineSeries({ color: '#2ecc71', lineWidth: 3, lineStyle: 0, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const fibMidSeries = chart.addLineSeries({ color: '#bfc7d5', lineWidth: 2, lineStyle: 0, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const fibShortSeries = chart.addLineSeries({ color: '#f7768e', lineWidth: 3, lineStyle: 0, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
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
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width);
      canvas.height = Math.floor(rect.height);
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#181c24';
      ctx.fillRect(0, 0, w, h);

      const trades = latestPayload.equity_curve || [];
      if (!trades.length) {
        ctx.fillStyle = '#8e98bf';
        ctx.font = '11px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('No closed trades', w / 2, h / 2);
        return;
      }

      const vals = trades.map(t => t.cum_pct);
      const minV = Math.min(0, ...vals);
      const maxV = Math.max(0, ...vals);
      const range = Math.max(1, maxV - minV) * 1.15;
      const padL = 8, padR = 8, padT = 14, padB = 14;
      const pw = w - padL - padR, ph = h - padT - padB;

      function tx(i) { return padL + (i / Math.max(1, trades.length - 1)) * pw; }
      function ty(v) { return padT + (1 - (v - minV) / range) * ph; }

      // zero line
      const y0 = ty(0);
      ctx.strokeStyle = '#3a3f52';
      ctx.lineWidth = 0.7;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(padL, y0); ctx.lineTo(w - padR, y0); ctx.stroke();
      ctx.setLineDash([]);

      // step line
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < trades.length; i++) {
        const x = tx(i);
        const y = ty(trades[i].cum_pct);
        if (i === 0) { ctx.moveTo(x, y0); ctx.lineTo(x, y); }
        else {
          ctx.lineTo(x, ty(trades[i - 1].cum_pct));
          ctx.lineTo(x, y);
        }
      }
      const lastCum = trades[trades.length - 1].cum_pct;
      ctx.strokeStyle = lastCum >= 0 ? '#9ece6a' : '#f7768e';
      ctx.stroke();

      // % labels at each step
      ctx.font = 'bold 9px system-ui';
      ctx.textAlign = 'center';
      for (let i = 0; i < trades.length; i++) {
        const x = tx(i);
        const y = ty(trades[i].cum_pct);
        const pnl = trades[i].pnl_pct;
        ctx.fillStyle = pnl >= 0 ? '#9ece6a' : '#f7768e';
        const label = (pnl >= 0 ? '+' : '') + pnl.toFixed(1) + '%';
        ctx.fillText(label, x, y - 5);
      }

      // cumulative at the end
      ctx.font = 'bold 11px system-ui';
      ctx.textAlign = 'right';
      ctx.fillStyle = lastCum >= 0 ? '#9ece6a' : '#f7768e';
      ctx.fillText((lastCum >= 0 ? '+' : '') + lastCum.toFixed(1) + '%', w - padR, padT - 2);
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
      document.getElementById('position').textContent = pos.position != null ? String(pos.position) : (pos.error || '-');
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
      const payload = await fetch('/api/dashboard/chart?hours=336&max_points=4000').then(r => r.json());
      latestPayload = payload;
      if (!payload.ok) return;

      const barsRaw = Array.isArray(payload.bars) ? payload.bars : [];
      timeMap = buildTimeMapFromBars(barsRaw);
      const bars = mapBarsForChart(barsRaw);
      candle.setData(bars);
      candle.setMarkers(mapMarkersForChart(Array.isArray(payload.markers) ? payload.markers : []));

      barsRawRef = barsRaw;

      const levels = payload.levels || {};
      const exitInfo = buildUnifiedExitLine(bars, levels);
      slSeries.setData(levelLineFromEntry(bars, levels.sl, levels));
      ttpSeries.setData(levelLineFromEntry(bars, levels.ttp, levels));

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

      for (const s of tradeSegmentSeries) chart.removeSeries(s);
      tradeSegmentSeries.length = 0;
      const segments = Array.isArray(payload.segments) ? payload.segments : [];
      for (const seg of segments) {
        const ls = chart.addLineSeries({ color: seg.color || '#9aa5b1', lineWidth: 2, title: seg.positive ? 'Trade +' : 'Trade -' });
        ls.setData(mapSegmentForChart(seg));
        tradeSegmentSeries.push(ls);
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

    chart.timeScale().subscribeVisibleTimeRangeChange(() => drawRegimeBand());
    window.addEventListener('resize', () => { drawRegimeBand(); drawAllHeatmaps(); drawEquityCurve(); });
    document.getElementById('traj-window').addEventListener('change', loadStateSpace);

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
      await Promise.all([loadMeta(), loadChart()]);
    }
    tick();
    loadStateSpace();
    setInterval(tick, 10000);
    setInterval(loadStateSpace, 30000);
  </script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """Simple dashboard UI: API status, SOL ticker from KuCoin, position. Basis für spätere Desktop-App."""
    return DASHBOARD_HTML


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
    # Railway/cloud set PORT; use it so the app listens on the right port
    port = int(os.environ.get("PORT", str(args.port)))
    uvicorn.run(
        "quant.execution.webhook_server:app",
        host=args.host,
        port=port,
        reload=bool(args.reload),
        log_level="info",
    )
