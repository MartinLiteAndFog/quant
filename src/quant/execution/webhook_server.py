# src/quant/execution/webhook_server.py

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
import uvicorn

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


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quant Live Dashboard</title>
  <style>
    :root { --bg: #1a1b26; --card: #24283b; --text: #c0caf5; --accent: #7aa2f7; --ok: #9ece6a; --err: #f7768e; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 1rem; }
    h1 { font-size: 1.25rem; color: var(--accent); }
    .card { background: var(--card); border-radius: 8px; padding: 1rem; margin: 1rem 0; max-width: 480px; }
    .row { display: flex; justify-content: space-between; margin: 0.5rem 0; }
    .label { color: #787c99; }
    .ok { color: var(--ok); }
    .err { color: var(--err); }
    .hint { font-size: 0.875rem; color: #787c99; margin-top: 0.5rem; }
    a { color: var(--accent); }
  </style>
</head>
<body>
  <h1>Quant Live Dashboard</h1>
  <div class="card">
    <div class="row"><span class="label">API (KuCoin)</span><span id="api-status">…</span></div>
    <div class="row"><span class="label">Kurs (Bid / Ask)</span><span id="ticker">…</span></div>
    <div class="row"><span class="label">Position</span><span id="position">…</span></div>
    <p class="hint" id="hint"></p>
  </div>
  <p class="hint">API-Key: .env oder Umgebungsvariablen (KUCOIN_FUTURES_*). Siehe docs/LIVE_DEPLOY.md</p>
  <script>
    async function load() {
      const [st, pos] = await Promise.all([
        fetch('/api/status').then(r => r.json()),
        fetch('/api/position').then(r => r.json())
      ]);
      document.getElementById('api-status').textContent = st.api_configured ? '✓ konfiguriert' : '— nicht gesetzt';
      document.getElementById('api-status').className = st.api_configured ? 'ok' : 'err';
      if (st.hint) document.getElementById('hint').textContent = st.hint;
      if (st.ticker) {
        const t = st.ticker;
        const b = typeof t.bid === 'number' ? t.bid.toFixed(4) : t.bid;
        const a = typeof t.ask === 'number' ? t.ask.toFixed(4) : t.ask;
        const m = t.mid != null ? (typeof t.mid === 'number' ? t.mid.toFixed(4) : t.mid) : null;
        document.getElementById('ticker').textContent = m != null ? m + ' (bid ' + b + ' / ask ' + a + ')' : (b + ' / ' + a);
      } else document.getElementById('ticker').textContent = st.ticker_error || '—';
      document.getElementById('position').textContent = pos.position != null ? pos.position : (pos.error || '—');
    }
    load();
    setInterval(load, 10000);
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
    # Railway/cloud set PORT; use it so the app listens on the right port
    port = int(os.environ.get("PORT", str(args.port)))
    uvicorn.run(
        "quant.execution.webhook_server:app",
        host=args.host,
        port=port,
        reload=bool(args.reload),
        log_level="info",
    )
