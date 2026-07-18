from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from quant.execution.bot_profiles import (
    PROFILE_CANONICAL,
    PROFILE_COUNTERTREND,
    PROFILE_COUNTERTREND_SL_REVERSE,
    SUPPORTED_PROFILES,
    active_profile,
)


_BACKTEST_DEFAULTS = {
    "LIVE_IMBA_LOOKBACK": "150",
    "LIVE_FLIP_TTP_TRAIL_PCT": "0.0025",
    "LIVE_FLIP_MIN_SL_PCT": "0.010",
    "LIVE_FLIP_MAX_SL_PCT": "0.080",
    "LIVE_FLIP_SWING_LOOKBACK": "180",
}

_MICRO_PILOT_DEFAULTS = {
    # No absolute dollar or contract cap by default — position size is governed
    # purely by LIVE_EXECUTOR_POS_PCT (percentage of equity). Set
    # LIVE_EXECUTOR_MAX_MARGIN_USDT / _MAX_CONTRACTS explicitly to add a hard
    # backstop on top of that.
    "LIVE_EXECUTOR_MAX_LEVERAGE": "3",
    "LIVE_EXECUTOR_LEVERAGE": "3",
    "KUCOIN_FUTURES_ORDER_LEVERAGE": "3",
    "KUCOIN_FUTURES_MARGIN_MODE": "isolated",
    "KUCOIN_FUTURES_STRICT_MARGIN_MODE": "1",
    "LIVE_EXECUTOR_POS_PCT": "0.90",
}

# Absolute ceilings the pilot refuses to exceed. These are the guardrails, not
# the operating values — raise them deliberately, per account. KuCoin itself
# allows up to 75x on SOL-USDT, so these are our limits, not the exchange's.
_PILOT_CEILINGS = {
    "leverage": ("MICRO_PILOT_LEVERAGE_CEILING", 10.0),
    "margin_usdt": ("MICRO_PILOT_MARGIN_CEILING_USDT", 20.0),
    "contracts": ("MICRO_PILOT_CONTRACTS_CEILING", 50.0),
}


def _ceiling(name: str) -> float:
    env_key, default = _PILOT_CEILINGS[name]
    return float(os.getenv(env_key, str(default)))


def _safe_instance_id(raw: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw.strip().lower())
    value = re.sub(r"[-_]+", "-", value).strip("-")
    if not value:
        raise ValueError("BOT_INSTANCE_ID must contain at least one letter or number")
    return value


def configure_environment() -> tuple[str, str]:
    profile = active_profile()
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"BOT_PROFILE is required and must be one of {sorted(SUPPORTED_PROFILES)}")
    instance = _safe_instance_id(os.getenv("BOT_INSTANCE_ID", profile))
    root = Path(os.getenv("BOT_DATA_ROOT", "/data/live/bots")) / instance

    os.environ["BOT_INSTANCE_ID"] = instance
    os.environ.setdefault("SIGNALS_DIR", str(root / "signals"))
    os.environ.setdefault("LIVE_SIGNAL_STATE", str(root / "live_signal_state.json"))
    os.environ.setdefault("LIVE_EXECUTOR_STATE", str(root / "live_executor_state.json"))
    os.environ.setdefault("EVENTS_DIR", str(root / "events"))
    if profile in {PROFILE_COUNTERTREND, PROFILE_COUNTERTREND_SL_REVERSE}:
        for key, value in _BACKTEST_DEFAULTS.items():
            os.environ.setdefault(key, value)
    if str(os.getenv("MICRO_PILOT_MODE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        for key, value in _MICRO_PILOT_DEFAULTS.items():
            os.environ.setdefault(key, value)
        leverage = float(os.environ["LIVE_EXECUTOR_LEVERAGE"])
        order_leverage = float(os.environ["KUCOIN_FUTURES_ORDER_LEVERAGE"])
        max_leverage = float(os.environ["LIVE_EXECUTOR_MAX_LEVERAGE"])
        # 0 / unset means "no absolute cap" — sizing is governed by
        # LIVE_EXECUTOR_POS_PCT alone. Only validate a cap that was asked for.
        max_margin = float(os.getenv("LIVE_EXECUTOR_MAX_MARGIN_USDT", "0") or 0)
        max_contracts = int(float(os.getenv("LIVE_EXECUTOR_MAX_CONTRACTS", "0") or 0))
        pos_pct = float(os.environ["LIVE_EXECUTOR_POS_PCT"])
        margin_mode = os.environ["KUCOIN_FUTURES_MARGIN_MODE"].strip().lower()
        if leverage <= 0 or leverage > max_leverage or order_leverage != leverage:
            raise ValueError("micro pilot requires matching executor/order leverage within the configured cap")
        if max_leverage > _ceiling("leverage"):
            raise ValueError(
                f"micro pilot leverage cap {max_leverage} exceeds ceiling {_ceiling('leverage')}"
            )
        if not 0 < pos_pct <= 1:
            raise ValueError(f"LIVE_EXECUTOR_POS_PCT must be in (0, 1], got {pos_pct}")
        if max_margin > _ceiling("margin_usdt"):
            raise ValueError(
                f"micro pilot margin cap {max_margin} exceeds ceiling {_ceiling('margin_usdt')} USDT"
            )
        if max_contracts > _ceiling("contracts"):
            raise ValueError(
                f"micro pilot contract cap {max_contracts} exceeds ceiling {int(_ceiling('contracts'))}"
            )
        if margin_mode != "isolated":
            raise ValueError("micro pilot requires isolated margin")

    if tv_webhook_mode():
        # TV_EXEC_* defaults (pos_pct 0.50, leverage 10) are unrelated to this
        # bot's configured risk. Inherit the live executor settings so the
        # micro-pilot caps actually bind on the webhook path too.
        os.environ.setdefault("TV_EXEC_POS_PCT", os.getenv("LIVE_EXECUTOR_POS_PCT", "1.0"))
        os.environ.setdefault("TV_EXEC_LEVERAGE", os.getenv("LIVE_EXECUTOR_LEVERAGE", "1"))
        os.environ.setdefault(
            "TV_EXEC_DRY_RUN",
            "0" if os.getenv("LIVE_EXECUTOR_DRY_RUN", "1").strip() == "0" else "1",
        )
        tv_leverage = float(os.environ["TV_EXEC_LEVERAGE"])
        max_leverage = float(os.getenv("LIVE_EXECUTOR_MAX_LEVERAGE", "0") or 0)
        if max_leverage > 0 and tv_leverage > max_leverage:
            raise ValueError(
                f"TV_EXEC_LEVERAGE={tv_leverage} exceeds LIVE_EXECUTOR_MAX_LEVERAGE={max_leverage}"
            )

    return profile, instance


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def tv_webhook_mode() -> bool:
    """TradingView-driven mode: webhook is the sole source of buy/sell signals."""
    return _truthy(os.getenv("TV_WEBHOOK_ENABLED"))


def main() -> None:
    profile, instance = configure_environment()
    symbol = os.getenv("LIVE_SYMBOL", "SOL-USDT")
    signals_dir = os.environ["SIGNALS_DIR"]

    if tv_webhook_mode():
        # TradingView replaces the internal Renko signal worker entirely.
        # tv_signal_executor places orders directly, so live_executor is not
        # started either — two controllers on one sub-account would fight.
        os.environ.setdefault("ENABLE_TV_EXECUTOR", "1")
        commands = [
            [sys.executable, "-u", "-m", "quant.execution.bot_webhook"],
        ]
    else:
        commands = [
            [
                sys.executable,
                "-u",
                "-m",
                "quant.execution.live_signal_worker",
                "--symbol",
                symbol,
                "--signals-dir",
                signals_dir,
            ],
            [
                sys.executable,
                "-u",
                "-m",
                "quant.execution.live_executor",
                "--symbol",
                symbol,
                "--signals-dir",
                signals_dir,
            ],
        ]
    print(
        f"starting railway bot instance={instance} profile={profile} symbol={symbol} "
        f"signals_dir={signals_dir} tv_webhook_mode={tv_webhook_mode()} "
        f"supported_profiles={sorted(SUPPORTED_PROFILES)}",
        flush=True,
    )

    children = [subprocess.Popen(cmd, env=os.environ.copy()) for cmd in commands]

    def stop_children(signum: int, _frame: object) -> None:
        for child in children:
            if child.poll() is None:
                child.send_signal(signum)

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    exit_code = 0
    try:
        while True:
            stopped = [(child, child.poll()) for child in children if child.poll() is not None]
            if stopped:
                exit_code = next((int(code) for _, code in stopped if code), 0)
                stop_children(signal.SIGTERM, None)
                break
            time.sleep(0.5)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
