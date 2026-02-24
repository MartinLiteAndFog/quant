# src/quant/execution/webhook_server.py

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
import uvicorn

from quant.execution.dashboard_state import (
    build_fibo_levels,
    build_regime_overlay,
    load_active_levels,
    load_live_fill_markers,
    load_renko_bars,
    load_renko_health,
    load_trade_segments,
    load_trade_markers,
)
from quant.regime import RegimeStore
from ..utils.log import get_logger

log = get_logger("quant.webhook")

app = FastAPI(title="quant-webhook", version="0.1.0")

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
        out["hint"] = "Set KUCOIN_FUTURES_API_KEY, KUCOIN_FUTURES_API_SECRET, KUCOIN_FUTURES_PASSPHRASE (e.g. in .env or cloud env vars)."
        return out
    try:
        broker = _kucoin_broker()
        bid, ask = broker.get_best_bid_ask(symbol)
        out["ticker"] = {"symbol": symbol, "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0 if (bid and ask) else None}
    except Exception as e:
        out["ticker"] = None
        out["ticker_error"] = str(e)
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
        markers = load_trade_markers(max_points=int(max(100, max_points)))
        markers_live = load_live_fill_markers(symbol=symbol, limit=int(max(50, min(500, max_points))))
        markers = sorted(markers + markers_live, key=lambda x: int(x.get("time", 0)))
        if len(markers) > int(max(100, max_points)):
            markers = markers[-int(max(100, max_points)) :]
        segments = load_trade_segments(max_points=int(max(100, max_points)))
        levels = load_active_levels()
        regime = build_regime_overlay(symbol=symbol, hours=int(max(1, hours)))
        fibo = build_fibo_levels(max_points=int(max(100, max_points)))
        renko_health = load_renko_health()
        latest = regime.get("latest") or {}
        return {
            "ok": True,
            "symbol": symbol,
            "bars": bars,
            "markers": markers,
            "segments": segments,
            "levels": levels,
            "fibo": fibo,
            "renko_health": renko_health,
            "regime": regime,
            "confidence": latest.get("confidence"),
            "gate_on": latest.get("gate_on"),
            "regime_state": latest.get("regime_state"),
            "ts": _now_utc_iso(),
        }
    except Exception as e:
        return {
            "ok": False,
            "symbol": symbol,
            "bars": [],
            "markers": [],
            "segments": [],
            "levels": {},
            "regime": {"spans": [], "points": [], "latest": None},
            "fibo": {"lookback": None, "long": [], "mid": [], "short": [], "latest": {}},
            "renko_health": {"ok": False, "bars": 0, "last_ts": None, "age_sec": None},
            "error": str(e),
            "ts": _now_utc_iso(),
        }


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
    #chart { position: absolute; inset: 0; z-index: 1; }
    #shade { position: absolute; inset: 0; pointer-events: none; z-index: 3; }
    .row { display: flex; justify-content: space-between; gap: 1rem; margin: 0.4rem 0; }
    .label { color: var(--muted); }
    .ok { color: var(--ok); }
    .err { color: var(--err); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .hint { color: var(--muted); font-size: 0.85rem; margin-top: 0.5rem; }
    .confidence-pill { font-weight: 700; }
    @media (max-width: 1200px) { .layout { grid-template-columns: 1fr; } .chart-wrap { height: 520px; } }
  </style>
</head>
<body>
  <h1>Quant Live Dashboard</h1>
  <div class="layout">
    <div class="card">
      <div class="chart-wrap">
        <div id="chart"></div>
        <canvas id="shade"></canvas>
      </div>
      <div class="hint">Gate ON is green, Gate OFF is blue. Intensity follows confidence.</div>
    </div>
    <div class="card">
      <div class="row"><span class="label">API (KuCoin)</span><span id="api-status">...</span></div>
      <div class="row"><span class="label">Ticker</span><span id="ticker" class="mono">...</span></div>
      <div class="row"><span class="label">Position (contracts)</span><span id="position" class="mono">...</span></div>
      <div class="row"><span class="label">Notional (est)</span><span id="position-notional" class="mono">...</span></div>
      <hr style="border-color:#2a3044;border-style:solid;border-width:1px 0 0 0;margin:0.8rem 0;">
      <div class="row"><span class="label">Gate</span><span id="gate">...</span></div>
      <div class="row"><span class="label">Regime</span><span id="regime-state">...</span></div>
      <div class="row"><span class="label">Confidence</span><span id="confidence" class="confidence-pill">...</span></div>
      <div class="row"><span class="label">Bar time</span><span id="bar-time" class="mono">-</span></div>
      <div class="row"><span class="label">SL active</span><span id="sl-active">-</span></div>
      <div class="row"><span class="label">TTP active</span><span id="ttp-active">-</span></div>
      <div class="row"><span class="label">SL</span><span id="lvl-sl" class="mono">-</span></div>
      <div class="row"><span class="label">TTP</span><span id="lvl-ttp" class="mono">-</span></div>
      <div class="row"><span class="label">TP1</span><span id="lvl-tp1" class="mono">-</span></div>
      <div class="row"><span class="label">TP2</span><span id="lvl-tp2" class="mono">-</span></div>
      <p id="hint" class="hint"></p>
    </div>
  </div>

  <script>
    const chartEl = document.getElementById('chart');
    const shadeCanvas = document.getElementById('shade');
    const qs = new URLSearchParams(window.location.search);
    const chartMode = (qs.get('mode') || 'brick').toLowerCase(); // brick | time
    const brickBaseTs = 1704067200; // 2024-01-01 UTC
    const chart = LightweightCharts.createChart(chartEl, {
      layout: { background: { color: '#1e2333' }, textColor: '#d9def7' },
      localization: {
        timeFormatter: (time) => {
          const t = Number(time);
          if (!Number.isFinite(t)) return '';
          if (chartMode !== 'brick' || !Array.isArray(barsRawRef) || !barsRawRef.length) {
            const d = new Date(t * 1000);
            const yyyy = d.getUTCFullYear();
            const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
            const dd = String(d.getUTCDate()).padStart(2, '0');
            const hh = String(d.getUTCHours()).padStart(2, '0');
            const mi = String(d.getUTCMinutes()).padStart(2, '0');
            return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
          }
          const idx = Math.max(0, Math.round((t - brickBaseTs) / 60));
          if (idx >= barsRawRef.length) return `B${idx}`;
          const rt = Number(barsRawRef[idx].time);
          if (!Number.isFinite(rt)) return `B${idx}`;
          const d = new Date(rt * 1000);
          const yyyy = d.getUTCFullYear();
          const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
          const dd = String(d.getUTCDate()).padStart(2, '0');
          const hh = String(d.getUTCHours()).padStart(2, '0');
          const mi = String(d.getUTCMinutes()).padStart(2, '0');
          return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
        },
      },
      rightPriceScale: { borderColor: '#2a3044' },
      timeScale: {
        borderColor: '#2a3044',
        timeVisible: chartMode !== 'brick',
        secondsVisible: false,
        // In brick mode, render a brick index instead of fake calendar time.
        tickMarkFormatter: (time, tickMarkType, locale) => {
          if (chartMode !== 'brick') return undefined;
          const t = Number(time);
          if (!Number.isFinite(t)) return '';
          const idx = Math.max(0, Math.round((t - brickBaseTs) / 60));
          if (!Array.isArray(barsRawRef) || idx >= barsRawRef.length) return `B${idx}`;
          const rt = Number(barsRawRef[idx].time);
          if (!Number.isFinite(rt)) return `B${idx}`;
          const d = new Date(rt * 1000);
          const hh = String(d.getUTCHours()).padStart(2, '0');
          const mm = String(d.getUTCMinutes()).padStart(2, '0');
          const yyyy = d.getUTCFullYear();
          const mon = String(d.getUTCMonth() + 1).padStart(2, '0');
          const day = String(d.getUTCDate()).padStart(2, '0');
          return `${day}.${mon} ${hh}:${mm}`;
        },
      },
      grid: { vertLines: { color: '#252b3f' }, horzLines: { color: '#252b3f' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
    });
    const candle = chart.addCandlestickSeries({
      upColor: '#2ecc71',
      downColor: '#f7768e',
      borderDownColor: '#f7768e',
      borderUpColor: '#2ecc71',
      wickDownColor: '#f7768e',
      wickUpColor: '#2ecc71',
    });
    const slSeries = chart.addLineSeries({ color: '#f7768e', lineWidth: 2, title: 'SL' });
    const ttpSeries = chart.addLineSeries({ color: '#ffcc66', lineWidth: 2, title: 'TTP' });
    const tp1Series = chart.addLineSeries({ color: '#7aa2f7', lineWidth: 2, title: 'TP1' });
    const tp2Series = chart.addLineSeries({ color: '#bb9af7', lineWidth: 2, title: 'TP2' });
    const fibLongSeries = chart.addLineSeries({ color: '#2ecc71', lineWidth: 3, lineStyle: 0, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const fibMidSeries = chart.addLineSeries({ color: '#bfc7d5', lineWidth: 2, lineStyle: 0, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const fibShortSeries = chart.addLineSeries({ color: '#f7768e', lineWidth: 3, lineStyle: 0, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const priceLineSeries = chart.addLineSeries({ color: '#9aa5b1', lineWidth: 1, title: 'Last', lineStyle: 2, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    const tradeSegmentSeries = [];

    let latestPayload = null;
    let timeMap = null;
    let barsRawRef = [];
    let latestMid = null;
    let hasFittedOnce = false;

    function resizeShade() {
      shadeCanvas.width = chartEl.clientWidth;
      shadeCanvas.height = chartEl.clientHeight;
    }

    function confAlpha(conf) {
      // Temporary strong colors requested by user; confidence can be reintroduced later.
      return 0.24;
    }

    function drawGateShading() {
      resizeShade();
      const ctx = shadeCanvas.getContext('2d');
      ctx.clearRect(0, 0, shadeCanvas.width, shadeCanvas.height);
      if (!latestPayload || !latestPayload.regime || !Array.isArray(latestPayload.regime.spans)) return;
      const spans = latestPayload.regime.spans;
      if (Array.isArray(spans) && spans.length === 1 && Number(spans[0].from) === Number(spans[0].to) && latestPayload?.bars?.length) {
        spans[0].from = Number(latestPayload.bars[0].time);
        spans[0].to = Number(latestPayload.bars[latestPayload.bars.length - 1].time);
      }
      const tscale = chart.timeScale();
      const chartH = chartEl.clientHeight;
      for (const s of spans) {
        const fromT = mapTimeForChart(s.from);
        const toT = mapTimeForChart(s.to);
        if (fromT == null || toT == null) continue;
        const x0 = tscale.timeToCoordinate(fromT);
        const x1 = tscale.timeToCoordinate(toT);
        if (x0 == null || x1 == null) continue;
        const left = Math.min(x0, x1);
        const width = Math.max(1, Math.abs(x1 - x0));
        const alpha = confAlpha(s.confidence);
        const color = Number(s.gate_on) ? `rgba(46, 204, 113, ${alpha})` : `rgba(64, 124, 255, ${alpha})`;
        ctx.fillStyle = color;
        ctx.fillRect(left, 0, width, chartH);
      }
    }

    function buildTimeMapFromBars(bars) {
      const m = new Map();
      if (!Array.isArray(bars)) return m;
      for (let i = 0; i < bars.length; i++) {
        const t = Number(bars[i].time);
        if (!Number.isFinite(t)) continue;
        // Keep first occurrence for stable shading/marker placement.
        if (!m.has(t)) m.set(t, i);
      }
      return m;
    }

    function mapTimeForChart(t) {
      if (t == null) return null;
      const n = Number(t);
      if (!Number.isFinite(n)) return null;
      if (chartMode !== 'brick' || !timeMap) return n;
      let idx = timeMap.get(n);
      if (idx == null && Array.isArray(barsRawRef) && barsRawRef.length) {
        // Nearest-left mapping for regime spans whose timestamps are not exact brick timestamps.
        let lo = 0, hi = barsRawRef.length - 1, best = null;
        while (lo <= hi) {
          const m = Math.floor((lo + hi) / 2);
          const tv = Number(barsRawRef[m].time);
          if (tv <= n) {
            best = m;
            lo = m + 1;
          } else {
            hi = m - 1;
          }
        }
        idx = best;
      }
      if (idx == null) return null;
      return brickBaseTs + idx * 60;
    }

    function mapBarsForChart(bars) {
      if (!Array.isArray(bars)) return [];
      if (chartMode !== 'brick') return bars;
      return bars.map((b, i) => ({
        ...b,
        time: brickBaseTs + i * 60,
      }));
    }

    function mapMarkersForChart(markers) {
      if (!Array.isArray(markers)) return [];
      if (chartMode !== 'brick') return markers;
      return markers
        .map((m) => {
          const mapped = mapTimeForChart(m.time);
          if (mapped == null) return null;
          return { ...m, time: mapped };
        })
        .filter(Boolean);
    }

    function mapLineForChart(points) {
      if (!Array.isArray(points)) return [];
      if (chartMode !== 'brick') return points;
      return points
        .map((p) => {
          const mapped = mapTimeForChart(p.time);
          if (mapped == null) return null;
          return { time: mapped, value: Number(p.value) };
        })
        .filter(Boolean);
    }

    function mapSegmentForChart(seg) {
      const t0 = mapTimeForChart(seg.from_time);
      const t1 = mapTimeForChart(seg.to_time);
      if (t0 == null || t1 == null) return [];
      return [
        { time: t0, value: Number(seg.from_price) },
        { time: t1, value: Number(seg.to_price) },
      ];
    }

    function fmtNum(v) {
      if (v == null || Number.isNaN(Number(v))) return '-';
      return Number(v).toFixed(4);
    }

    function levelLineData(lastBars, level) {
      if (!Array.isArray(lastBars) || !lastBars.length || level == null || Number.isNaN(Number(level))) return [];
      const first = lastBars[0].time;
      const last = lastBars[lastBars.length - 1].time;
      const val = Number(level);
      return [{ time: first, value: val }, { time: last, value: val }];
    }

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
        latestMid = Number(t.mid);
        document.getElementById('ticker').textContent = m != null ? `${m} (bid ${b} / ask ${a})` : `${b} / ${a}`;
      } else {
        latestMid = null;
        document.getElementById('ticker').textContent = st.ticker_error || '-';
      }
      document.getElementById('position').textContent = pos.position != null ? String(pos.position) : (pos.error || '-');
      if (pos.position != null && st.ticker && st.ticker.mid != null) {
        const notional = Math.abs(Number(pos.position)) * Number(st.ticker.mid);
        document.getElementById('position-notional').textContent = Number.isFinite(notional) ? `${notional.toFixed(2)} USDT` : '-';
      } else {
        document.getElementById('position-notional').textContent = '-';
      }
    }

    async function loadChart() {
      const payload = await fetch('/api/dashboard/chart?hours=336&max_points=4000').then(r => r.json());
      latestPayload = payload;
      if (!payload.ok) return;

      const barsRaw = Array.isArray(payload.bars) ? payload.bars : [];
      barsRawRef = barsRaw;
      timeMap = buildTimeMapFromBars(barsRaw);
      const bars = mapBarsForChart(barsRaw);
      candle.setData(bars);
      candle.setMarkers(mapMarkersForChart(Array.isArray(payload.markers) ? payload.markers : []));

      const levels = payload.levels || {};
      slSeries.setData(levelLineData(bars, levels.sl));
      ttpSeries.setData(levelLineData(bars, levels.ttp));
      tp1Series.setData(levelLineData(bars, levels.tp1));
      tp2Series.setData(levelLineData(bars, levels.tp2));
      const fibo = payload.fibo || {};
      fibLongSeries.setData(mapLineForChart(fibo.long || []));
      fibMidSeries.setData(mapLineForChart(fibo.mid || []));
      fibShortSeries.setData(mapLineForChart(fibo.short || []));
      const lastBar = bars.length ? bars[bars.length - 1] : null;
      if (lastBar) {
        const livePx = Number.isFinite(Number(latestMid)) ? Number(latestMid) : Number(lastBar.close);
        priceLineSeries.setData([
          { time: bars[0].time, value: livePx },
          { time: lastBar.time, value: livePx },
        ]);
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

      document.getElementById('lvl-sl').textContent = fmtNum(levels.sl);
      document.getElementById('lvl-ttp').textContent = fmtNum(levels.ttp);
      document.getElementById('lvl-tp1').textContent = fmtNum(levels.tp1);
      document.getElementById('lvl-tp2').textContent = fmtNum(levels.tp2);
      const slActive = levels.sl != null && !Number.isNaN(Number(levels.sl));
      const ttpActive = levels.ttp != null && !Number.isNaN(Number(levels.ttp));
      const slEl = document.getElementById('sl-active');
      const ttpEl = document.getElementById('ttp-active');
      slEl.textContent = slActive ? 'YES' : 'NO';
      ttpEl.textContent = ttpActive ? 'YES' : 'NO';
      slEl.className = slActive ? 'ok' : 'err';
      ttpEl.className = ttpActive ? 'ok' : 'err';

      const gate = payload.gate_on;
      document.getElementById('gate').textContent = gate == null ? '-' : (Number(gate) ? 'ON' : 'OFF');
      document.getElementById('gate').className = Number(gate) ? 'ok' : 'err';
      document.getElementById('regime-state').textContent = payload.regime_state || '-';
      const conf = payload.confidence == null ? null : Number(payload.confidence);
      document.getElementById('confidence').textContent = conf == null ? '-' : conf.toFixed(3);
      if (conf != null) {
        document.getElementById('confidence').style.color = conf >= 0.7 ? '#9ece6a' : (conf >= 0.5 ? '#e0af68' : '#f7768e');
      }

      const rh = payload.renko_health || {};
      if (rh.age_sec != null) {
        document.getElementById('hint').textContent = `Renko age: ${Math.round(Number(rh.age_sec))}s`;
      }

      if (!hasFittedOnce) {
        chart.timeScale().fitContent();
        hasFittedOnce = true;
      }
      drawGateShading();
    }

    chart.timeScale().subscribeVisibleTimeRangeChange(() => drawGateShading());
    chart.subscribeCrosshairMove((param) => {
      const t = param && param.time != null ? Number(param.time) : null;
      if (t == null || !Number.isFinite(t)) {
        document.getElementById('bar-time').textContent = '-';
        return;
      }
      if (chartMode !== 'brick') {
        document.getElementById('bar-time').textContent = new Date(t * 1000).toISOString().slice(11, 16);
        return;
      }
      const idx = Math.max(0, Math.round((t - brickBaseTs) / 60));
      if (!Array.isArray(barsRawRef) || idx >= barsRawRef.length) {
        document.getElementById('bar-time').textContent = '-';
        return;
      }
      const rt = Number(barsRawRef[idx].time);
      if (!Number.isFinite(rt)) {
        document.getElementById('bar-time').textContent = '-';
        return;
      }
      document.getElementById('bar-time').textContent = new Date(rt * 1000).toISOString().slice(11, 16);
    });
    window.addEventListener('resize', () => drawGateShading());

    async function tick() {
      await Promise.all([loadMeta(), loadChart()]);
    }
    tick();
    setInterval(tick, 10000);
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
