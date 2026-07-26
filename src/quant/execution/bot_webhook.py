# src/quant/execution/bot_webhook.py
"""
Per-bot TradingView webhook receiver.

Each Railway pilot service runs its own copy of this app on its own public
domain, bound to its own KuCoin sub-account credentials. TradingView alerts
posted to `/webhook/tv-execute` are the sole source of buy/sell for that bot —
the Renko `live_signal_worker` is not started in this mode.

Env:
  BOT_PROFILE            countertrend | countertrend_sl_reverse | pc3axis | canonical
  BOT_INSTANCE_ID        strategy_instance tag used in Postgres
  BOT_WEBHOOK_TOKEN      shared secret; required unless BOT_WEBHOOK_ALLOW_ANON=1
  PORT                   bind port (Railway injects this)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request

from quant.execution.bot_profiles import active_profile, display_name, strategy_instance_id
from quant.execution.tv_signal_executor import (
    TVExecConfig,
    execute_tv_signal,
    parse_tv_signal,
    start_tv_executor,
)
from quant.execution.tv_signal_executor import _ready as tv_ready
from quant.utils.log import get_logger

log = get_logger("quant.bot_webhook")


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


def _signal_code(action: str, side: str) -> int:
    """Map a TradingView action/side pair onto the signal_events `signal`
    column, which is a smallint constrained to (-1, 0, 1).

    Directional actions (entry, flip) carry the side: buy -> long (+1),
    sell -> short (-1). Reducing actions (exit, tp1, tp2, sl) target flat (0),
    as does anything unrecognised — an unparseable signal asserts no direction.
    """
    if action in ("entry", "flip"):
        if side == "buy":
            return 1
        if side == "sell":
            return -1
    return 0


_SIGNAL_SIDE_BY_CODE = {1: "long", -1: "short", 0: "flat"}


def _log_inbound_signal(payload: Dict[str, Any], *, disposition: str, detail: str = "") -> None:
    """Best-effort record of every inbound TradingView signal and how it was
    dispositioned. Signals that never reach the executor — bot_mismatch,
    not_ready, parse_error — leave no action/execution event, so without this
    they vanish silently. Written to signal_events, surfaced by /diag/timeline.
    TradingView signals are sparse, so the extra write is negligible.

    The raw action/side strings are kept in payload_json: the typed columns
    only model direction, so "which of tp1/tp2/sl fired" would otherwise be
    lost.
    """
    try:
        from quant.execution.event_store import insert_signal_event

        action = str(payload.get("action", "")).strip().lower()
        side = str(payload.get("side", "")).strip().lower()
        sym = ""
        for k in ("symbol", "ticker", "pair"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                sym = v.strip().replace("-", "")
                break
        if not sym:
            sym = os.getenv("LIVE_SYMBOL", "SOL-USDT").replace("-", "")
        inst = strategy_instance_id()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        code = _signal_code(action, side)
        insert_signal_event(
            {
                "event_id": f"tvin:{inst}:{ts}:{disposition}",
                "ts": ts,
                "seq": int(time.time() * 1000),
                "strategy": "tv_executor",
                "strategy_instance": inst,
                "symbol": sym,
                "venue": "kucoin",
                "signal": code,
                "signal_side": _SIGNAL_SIDE_BY_CODE[code],
                "signal_family": "tradingview",
                "signal_kind": action or "unknown",
                "source_type": "tv_webhook",
                "source_event_id": None,
                "position_before": None,
                "engine_mode_before": None,
                "payload_json": {
                    "kind": "inbound_signal",
                    "disposition": disposition,
                    "detail": detail,
                    "action": action,
                    "raw_side": side,
                    "bot_target": str(payload.get("bot", "")).strip(),
                    "profile": active_profile(),
                },
            }
        )
    except Exception as e:
        # Never let diagnostics logging interfere with signal execution — but do
        # not fail silently either: a swallowed schema error here is exactly why
        # signal_events sat empty fleet-wide while signals were arriving.
        print(
            f"[bot_webhook] inbound signal logging failed "
            f"(disposition={disposition}): {type(e).__name__}: {e}",
            flush=True,
        )


def _expected_token() -> str:
    return str(os.getenv("BOT_WEBHOOK_TOKEN", "")).strip()


def _check_token(supplied: Optional[str]) -> None:
    if _truthy(os.getenv("BOT_WEBHOOK_ALLOW_ANON")):
        return
    expected = _expected_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="BOT_WEBHOOK_TOKEN not configured; refusing unauthenticated webhook",
        )
    if str(supplied or "").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid webhook token")


def _enforce_margin_mode() -> None:
    """Bring the sub-account to the configured margin mode before trading.

    CROSS mode makes KuCoin ignore the per-order leverage, so a bot set to 10x
    quietly trades at whatever cross gives it. Only acts when flat.
    """
    want = str(os.getenv("KUCOIN_FUTURES_MARGIN_MODE", "")).strip()
    if not want:
        return
    try:
        from quant.execution.kucoin_futures import KucoinFuturesBroker

        broker = KucoinFuturesBroker()
        symbol = os.getenv("LIVE_SYMBOL", "SOL-USDT")
        result = broker.ensure_margin_mode(symbol, want)
        if result and result.upper() != want.upper():
            log.error(
                "margin mode is %s but %s is configured — configured leverage "
                "will NOT apply until this is corrected",
                result, want.upper(),
            )
    except Exception as e:
        log.warning("margin mode check skipped: %s", e)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # tv_signal_executor only arms itself when ENABLE_TV_EXECUTOR=1.
    os.environ.setdefault("ENABLE_TV_EXECUTOR", "1")
    _enforce_margin_mode()
    start_tv_executor()
    # Fleet curves need equity history on a fixed clock, not just /health-poll
    # side effects (audit 2026-07-22). Default ON for pilots; opt out with
    # FLEET_EQUITY_WRITER_ENABLED=0.
    from quant.execution.equity_snapshot_writer import start_equity_snapshot_writer

    start_equity_snapshot_writer(
        venue="kucoin",
        account=strategy_instance_id(),
        default_enabled=True,
    )
    # Read-only: persist confirmed Funding↔Futures ledger transfers alongside
    # equity snapshots so Fleet can calculate cashflow-corrected performance.
    from quant.execution.cashflow_sync import start_cashflow_sync

    start_cashflow_sync(
        venue="kucoin",
        account=strategy_instance_id(),
        default_enabled=True,
    )
    log.info(
        "bot webhook ready name=%s profile=%s instance=%s symbol=%s",
        display_name(),
        active_profile(),
        strategy_instance_id(),
        os.getenv("LIVE_SYMBOL", "SOL-USDT"),
    )
    yield


app = FastAPI(title="quant-bot-webhook", version="0.1.0", lifespan=_lifespan)

# Read-only KuCoin Futures diagnostics (/kucoin/diag|positions|fills|orders|account),
# token-guarded by BOT_WEBHOOK_TOKEN. Exposes real positions/fills the /health
# endpoint cannot show (equity + armed flags only).
from quant.execution.kucoin_diag import router as kucoin_diag_router
from quant.execution.signal_diag import router as signal_diag_router

app.include_router(kucoin_diag_router)
app.include_router(signal_diag_router)


@app.get("/health")
def health() -> Dict[str, Any]:
    from quant.execution.live_account import live_account_snapshot, trading_mode_from_env

    mode = trading_mode_from_env()
    # Prefer explicit TV dry-run for this process.
    dry = _truthy(os.getenv("TV_EXEC_DRY_RUN", "1"))
    live = _truthy(os.getenv("LIVE_TRADING_ENABLED"))
    out: Dict[str, Any] = {
        "ok": True,
        "name": display_name(),
        "profile": active_profile(),
        "instance": strategy_instance_id(),
        "symbol": os.getenv("LIVE_SYMBOL", "SOL-USDT"),
        "executor_ready": tv_ready.is_set(),
        "dry_run": dry,
        "live_trading_enabled": live and not dry,
        "venue": "kucoin",
    }
    acct = live_account_snapshot(
        prefer="kucoin",
        persist_account=strategy_instance_id(),
        use_cache=True,
    )
    if acct.get("ok"):
        out["equity"] = acct.get("equity")
        out["available"] = acct.get("available")
        out["margin"] = acct.get("margin")
        out["unrealised_pnl"] = acct.get("unrealised_pnl")
        out["currency"] = acct.get("currency")
        out["equity_source"] = acct.get("source")
    elif acct.get("error"):
        out["equity_error"] = acct.get("error")
    # Keep mode keys consistent even if TV_EXEC_DRY_RUN overrides trading_mode_from_env.
    out.setdefault("live_trading_enabled", mode.get("live_trading_enabled"))
    return out


@app.post("/webhook/tv-execute")
async def tv_execute(
    request: Request,
    x_webhook_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")

    # TradingView cannot set custom headers. Accept the token from the URL query
    # string as well as the body, so a Pine strategy's alert_message can stay
    # pure signal JSON and the secret never has to live inside the script.
    _check_token(
        x_webhook_token
        or request.query_params.get("token")
        or payload.get("token")
    )

    # Reject alerts addressed to a different bot, so one shared TradingView
    # alert template cannot accidentally fire every sub-account at once.
    # Checked before readiness: "not addressed to me" is a valid answer even
    # if this bot's executor is still warming up.
    target = str(payload.get("bot", "")).strip().lower()
    if target and target not in {strategy_instance_id().lower(), active_profile()}:
        _log_inbound_signal(payload, disposition="skipped_bot_mismatch", detail=f"target={target}")
        return {"ok": True, "skipped": "bot_mismatch", "instance": strategy_instance_id()}

    if not tv_ready.is_set():
        _log_inbound_signal(payload, disposition="rejected_not_ready")
        raise HTTPException(status_code=503, detail="tv_executor not ready")

    config = TVExecConfig.from_env()

    try:
        signal = parse_tv_signal(payload, default_symbol=config.symbol)
    except ValueError as e:
        _log_inbound_signal(payload, disposition="rejected_parse_error", detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))

    _log_inbound_signal(payload, disposition="accepted", detail=f"action={signal.action} side={signal.side}")

    import asyncio

    result = await asyncio.get_running_loop().run_in_executor(
        None, execute_tv_signal, signal, config
    )
    log.info(
        "tv signal executed instance=%s action=%s side=%s result=%s",
        strategy_instance_id(), signal.action, signal.side, result.get("ok"),
    )
    return {**result, "instance": strategy_instance_id(), "profile": active_profile()}


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
