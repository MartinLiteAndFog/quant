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
    "LIVE_EXECUTOR_MAX_MARGIN_USDT": "5",
    "LIVE_EXECUTOR_MAX_CONTRACTS": "1",
    "LIVE_EXECUTOR_MAX_LEVERAGE": "3",
    "LIVE_EXECUTOR_LEVERAGE": "3",
    "KUCOIN_FUTURES_ORDER_LEVERAGE": "3",
    "KUCOIN_FUTURES_MARGIN_MODE": "isolated",
    "KUCOIN_FUTURES_STRICT_MARGIN_MODE": "1",
    "LIVE_EXECUTOR_POS_PCT": "1.0",
}


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
        max_margin = float(os.environ["LIVE_EXECUTOR_MAX_MARGIN_USDT"])
        max_contracts = int(float(os.environ["LIVE_EXECUTOR_MAX_CONTRACTS"]))
        margin_mode = os.environ["KUCOIN_FUTURES_MARGIN_MODE"].strip().lower()
        if leverage <= 0 or leverage > max_leverage or order_leverage != leverage:
            raise ValueError("micro pilot requires matching executor/order leverage within the configured cap")
        if max_leverage > 3:
            raise ValueError("micro pilot leverage cap must not exceed 3")
        if max_margin <= 0 or max_margin > 5:
            raise ValueError("micro pilot margin cap must be in (0, 5] USDT")
        if max_contracts != 1:
            raise ValueError("SOL micro pilot requires exactly one contract maximum")
        if margin_mode != "isolated":
            raise ValueError("micro pilot requires isolated margin")
    return profile, instance


def main() -> None:
    profile, instance = configure_environment()
    symbol = os.getenv("LIVE_SYMBOL", "SOL-USDT")
    signals_dir = os.environ["SIGNALS_DIR"]

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
        f"signals_dir={signals_dir} supported_profiles={sorted(SUPPORTED_PROFILES)}",
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
