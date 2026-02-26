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
    build_regime_overlay,
    load_active_levels,
    load_renko_bars,
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


def _contract_multiplier(symbol: str) -> float:
    norm = "".join(ch for ch in str(symbol or "").upper() if ch.isalnum())
    v = os.getenv(f"CONTRACT_MULTIPLIER_{norm}")
    if v is None:
        v = os.getenv("CONTRACT_MULTIPLIER_DEFAULT", "1.0")
    try:
        m = float(v)
    except Exception:
        m = 1.0
    return m if m > 0 else 1.0


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
    """
    Current position from KuCoin Futures.

    Backwards compatible:
    - keeps `position` (signed: >0 long, <0 short)
    Adds:
    - side, size_abs, leverage, entry_price, mark_price, margin_mode, contract
    """
    key = (os.getenv("KUCOIN_FUTURES_API_KEY") or "").strip()
    if not key:
        return {
            "ok": True,
            "symbol": symbol,
            "position": None,
            "contract_multiplier": _contract_multiplier(symbol),
            "hint": "Configure KuCoin API keys.",
        }
    try:
        broker = _kucoin_broker()
        # Prefer the raw position payload so we can expose leverage/entry/mark/margin mode.
        from quant.execution.kucoin_futures import _symbol_to_contract

        contract = _symbol_to_contract(symbol)
        data = broker._req("GET", f"/api/v1/position?symbol={contract}")
        if not isinstance(data, dict) or not data:
            return {
                "ok": True,
                "symbol": symbol,
                "contract": contract,
                "position": 0.0,
                "side": "flat",
                "size_abs": 0.0,
                "contract_multiplier": _contract_multiplier(symbol),
            }

        side = (data.get("side") or "").strip().lower() or "unknown"
        qty = float(data.get("currentQty", data.get("size", 0)) or 0.0)
        if side == "short":
            pos = -abs(qty)
        elif side == "long":
            pos = abs(qty)
        else:
            pos = float(qty)

        lev = data.get("leverage")
        if lev is None:
            lev = data.get("realLeverage")
        if lev is None:
            lev = data.get("initLeverage")

        mm = (data.get("marginMode") or data.get("margin_mode") or "").strip().upper()
        if mm not in ("CROSS", "ISOLATED"):
            cm = data.get("crossMode")
            if isinstance(cm, bool):
                mm = "CROSS" if cm else "ISOLATED"
            else:
                mm = None

        entry = data.get("avgEntryPrice") or data.get("entryPrice")
        mark = data.get("markPrice")

        return {
            "ok": True,
            "symbol": symbol,
            "contract": contract,
            "position": float(pos),
            "side": ("long" if pos > 0 else "short" if pos < 0 else "flat"),
            "size_abs": float(abs(pos)),
            "leverage": lev,
            "entry_price": entry,
            "mark_price": mark,
            "margin_mode": mm,
            "contract_multiplier": _contract_multiplier(symbol),
        }
    except Exception as e:
        return {
            "ok": False,
            "symbol": symbol,
            "position": None,
            "contract_multiplier": _contract_multiplier(symbol),
            "error": str(e),
        }


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
        levels = load_active_levels()
        regime = build_regime_overlay(symbol=symbol, hours=int(max(1, hours)))
        latest = regime.get("latest") or {}
        return {
            "ok": True,
            "symbol": symbol,
            "bars": bars,
            "markers": markers,
            "levels": levels,
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
            "levels": {},
            "regime": {"spans": [], "points": [], "latest": None},
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
    #chart { position: absolute; inset: 0; }
    #shade { position: absolute; inset: 0; pointer-events: none; }
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
      <div class="row"><span class="label">Position</span><span id="position" class="mono">...</span></div>
      <hr style="border-color:#2a3044;border-style:solid;border-width:1px 0 0 0;margin:0.8rem 0;">
      <div class="row"><span class="label">Gate</span><span id="gate">...</span></div>
      <div class="row"><span class="label">Regime</span><span id="regime-state">...</span></div>
      <div class="row"><span class="label">Confidence</span><span id="confidence" class="confidence-pill">...</span></div>
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
          return `B${idx}`;
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

    let latestPayload = null;
    let timeMap = null;

    function resizeShade() {
      shadeCanvas.width = chartEl.clientWidth;
      shadeCanvas.height = chartEl.clientHeight;
    }

    function confAlpha(conf) {
      const c = Math.max(0, Math.min(1, Number(conf || 0)));
      return 0.08 + 0.32 * c;
    }

    function drawGateShading() {
      resizeShade();
      const ctx = shadeCanvas.getContext('2d');
      ctx.clearRect(0, 0, shadeCanvas.width, shadeCanvas.height);
      if (!latestPayload || !latestPayload.regime || !Array.isArray(latestPayload.regime.spans)) return;
      const spans = latestPayload.regime.spans;
      const tscale = chart.timeScale();
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
        ctx.fillRect(left, 0, width, shadeCanvas.height);
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
      const idx = timeMap.get(n);
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
        document.getElementById('ticker').textContent = m != null ? `${m} (bid ${b} / ask ${a})` : `${b} / ${a}`;
      } else {
        document.getElementById('ticker').textContent = st.ticker_error || '-';
      }
      document.getElementById('position').textContent = pos.position != null ? String(pos.position) : (pos.error || '-');
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

      const levels = payload.levels || {};
      slSeries.setData(levelLineData(bars, levels.sl));
      ttpSeries.setData(levelLineData(bars, levels.ttp));
      tp1Series.setData(levelLineData(bars, levels.tp1));
      tp2Series.setData(levelLineData(bars, levels.tp2));

      document.getElementById('lvl-sl').textContent = fmtNum(levels.sl);
      document.getElementById('lvl-ttp').textContent = fmtNum(levels.ttp);
      document.getElementById('lvl-tp1').textContent = fmtNum(levels.tp1);
      document.getElementById('lvl-tp2').textContent = fmtNum(levels.tp2);

      const gate = payload.gate_on;
      document.getElementById('gate').textContent = gate == null ? '-' : (Number(gate) ? 'ON' : 'OFF');
      document.getElementById('gate').className = Number(gate) ? 'ok' : 'err';
      document.getElementById('regime-state').textContent = payload.regime_state || '-';
      const conf = payload.confidence == null ? null : Number(payload.confidence);
      document.getElementById('confidence').textContent = conf == null ? '-' : conf.toFixed(3);
      if (conf != null) {
        document.getElementById('confidence').style.color = conf >= 0.7 ? '#9ece6a' : (conf >= 0.5 ? '#e0af68' : '#f7768e');
      }

      chart.timeScale().fitContent();
      drawGateShading();
    }

    chart.timeScale().subscribeVisibleTimeRangeChange(() => drawGateShading());
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
