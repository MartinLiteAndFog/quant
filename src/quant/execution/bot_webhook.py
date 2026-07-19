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
from contextlib import asynccontextmanager
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
    log.info(
        "bot webhook ready name=%s profile=%s instance=%s symbol=%s",
        display_name(),
        active_profile(),
        strategy_instance_id(),
        os.getenv("LIVE_SYMBOL", "SOL-USDT"),
    )
    yield


app = FastAPI(title="quant-bot-webhook", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "name": display_name(),
        "profile": active_profile(),
        "instance": strategy_instance_id(),
        "symbol": os.getenv("LIVE_SYMBOL", "SOL-USDT"),
        "executor_ready": tv_ready.is_set(),
        "dry_run": _truthy(os.getenv("TV_EXEC_DRY_RUN", "1")),
        "live_trading_enabled": _truthy(os.getenv("LIVE_TRADING_ENABLED")),
    }


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
        return {"ok": True, "skipped": "bot_mismatch", "instance": strategy_instance_id()}

    if not tv_ready.is_set():
        raise HTTPException(status_code=503, detail="tv_executor not ready")

    config = TVExecConfig.from_env()

    try:
        signal = parse_tv_signal(payload, default_symbol=config.symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
