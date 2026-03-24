from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _live_default(rel_path: str) -> str:
    if Path("/data").exists():
        return str(Path("/data") / rel_path)
    return str(Path("data") / rel_path)


def _env_bool(name: str, default: bool = False) -> bool:
    v = str(os.getenv(name, str(default))).strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()


def _choppiness(df: pd.DataFrame, n: int) -> pd.Series:
    tr = _true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = pd.to_numeric(df["high"], errors="coerce").astype(float).rolling(n, min_periods=n).max()
    ll = pd.to_numeric(df["low"], errors="coerce").astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))


def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)

    up = high.diff()
    down = -low.diff()

    dm_plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = _true_range(df)
    atr = _wilder_smooth(tr, n)
    sm_plus = _wilder_smooth(dm_plus, n)
    sm_minus = _wilder_smooth(dm_minus, n)

    di_plus = 100.0 * (sm_plus / atr.replace(0.0, np.nan))
    di_minus = 100.0 * (sm_minus / atr.replace(0.0, np.nan))

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan)
    return _wilder_smooth(dx, n)


def _efficiency_ratio(df: pd.DataFrame, n: int) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    net = (close - close.shift(n)).abs()
    denom = close.diff().abs().rolling(n, min_periods=n).sum()
    return (net / denom.replace(0.0, np.nan)).clip(0.0, 1.0)


def _quantile_threshold(x: pd.Series, q: float) -> float:
    s = pd.to_numeric(x, errors="coerce").astype(float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float("nan")
    q = max(0.0, min(1.0, float(q)))
    return float(s.quantile(q))


def _hysteresis_high_is_good_last(x: pd.Series, on_th: float, off_th: float) -> tuple[int, int]:
    state = False
    last_val = np.nan
    for raw in x.values:
        v = float(raw) if pd.notna(raw) else np.nan
        last_val = v
        if np.isnan(v):
            continue
        if (not state) and v >= on_th:
            state = True
        elif state and v <= off_th:
            state = False
    return int(state), int(not state)


def _hysteresis_low_is_good_last(x: pd.Series, on_th: float, off_th: float, start_on: bool = True) -> tuple[int, int]:
    state = bool(start_on)
    for raw in x.values:
        v = float(raw) if pd.notna(raw) else np.nan
        if np.isnan(v):
            continue
        if state and v >= off_th:
            state = False
        elif (not state) and v <= on_th:
            state = True
    return int(state), int(not state)


def _publish_gate_state_to_redis(state: Dict[str, Any]) -> None:
    redis_url = str(os.getenv("REDIS_URL", "")).strip()
    if not redis_url:
        return
    try:
        import redis as redis_lib

        symbol = str(os.getenv("LIVE_GATE_SYMBOL", os.getenv("LIVE_SYMBOL", "SOL-USDT"))).strip().upper()
        canon = "".join(ch for ch in symbol if ch.isalnum())
        key = f"gate:{canon}:latest"

        r = redis_lib.from_url(redis_url, decode_responses=True)
        payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)
        r.set(key, payload)
    except Exception:
        return


def _load_live_renko() -> pd.DataFrame:
    renko_path = Path(
        os.getenv(
            "LIVE_RENKO_PATH",
            os.getenv(
                "DASHBOARD_RENKO_PARQUET",
                _live_default("live/renko_latest.parquet"),
            ),
        )
    )

    if not renko_path.exists():
        raise FileNotFoundError(f"missing_renko:{renko_path}")

    df = pd.read_parquet(renko_path).copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        df = df.reset_index().rename(columns={"index": "ts"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    need = {"ts", "open", "high", "low", "close"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"renko_missing_cols:{missing}")

    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        raise ValueError("renko_empty")

    return df


def _fit_thresholds_live(hist: pd.DataFrame) -> Dict[str, float]:
    mode = str(os.getenv("LIVE_GATE_MODE", "rolling_quantiles")).strip().lower()

    if mode == "fixed":
        return {
            "chop_on": _env_float("LIVE_GATE_CHOP_ON", 58.0),
            "chop_off": _env_float("LIVE_GATE_CHOP_OFF", 52.0),
            "adx_on": _env_float("LIVE_GATE_ADX_ON", 18.0),
            "adx_off": _env_float("LIVE_GATE_ADX_OFF", 25.0),
            "er_on": _env_float("LIVE_GATE_ER_ON", 0.30),
            "er_off": _env_float("LIVE_GATE_ER_OFF", 0.40),
        }

    # rolling_quantiles:
    # Settings reference from the historical filename:
    # qch=0.4 qadx=0.6 qer=0.3
    # Hysteresis band is configurable separately.
    q_band = _env_float("LIVE_GATE_Q_BAND", 0.05)

    qch = _env_float("LIVE_GATE_CHOP_Q", 0.40)
    qadx = _env_float("LIVE_GATE_ADX_Q", 0.60)
    qer = _env_float("LIVE_GATE_ER_Q", 0.30)

    chop_on_q = min(1.0, max(0.0, qch))
    chop_off_q = min(1.0, max(0.0, qch - q_band))

    adx_on_q = min(1.0, max(0.0, qadx))
    adx_off_q = min(1.0, max(0.0, qadx + q_band))

    er_on_q = min(1.0, max(0.0, qer))
    er_off_q = min(1.0, max(0.0, qer + q_band))

    return {
        "chop_on": _quantile_threshold(hist["CHOP"], chop_on_q),
        "chop_off": _quantile_threshold(hist["CHOP"], chop_off_q),
        "adx_on": _quantile_threshold(hist["ADX"], adx_on_q),
        "adx_off": _quantile_threshold(hist["ADX"], adx_off_q),
        "er_on": _quantile_threshold(hist["ER"], er_on_q),
        "er_off": _quantile_threshold(hist["ER"], er_off_q),
    }


def get_live_gate_state() -> Dict[str, Any]:
    """
    Live CHOP/ADX/ER gate provider based on live Renko bars.

    This replaces the old XYZ / PC statespace provider entirely.

    Primary behavior:
    - Compute CHOP / ADX / ER on live Renko.
    - Fit thresholds from recent live history (rolling quantiles) OR use fixed thresholds.
    - Build a 2-of-3 gate.
    - Expose both countertrend/trend views, while gate_on/gate_off follow LIVE_GATE_PRIMARY.

    Defaults:
    - LIVE_GATE_MODE=rolling_quantiles
    - LIVE_GATE_CHOP_Q=0.40
    - LIVE_GATE_ADX_Q=0.60
    - LIVE_GATE_ER_Q=0.30
    - LIVE_GATE_PRIMARY=off   # because your reference file was ...daily_OFF.csv

    If you want classic countertrend semantics as gate_on:
    - set LIVE_GATE_PRIMARY=on
    """
    try:
        bars = _load_live_renko()
    except Exception as e:
        return {
            "ts": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gate_on": 0,
            "gate_off": 1,
            "source": "default_off",
            "error": str(e),
        }

    max_age_sec = _env_float("LIVE_GATE_MAX_AGE_SEC", 1800.0)
    last_ts = pd.Timestamp(bars["ts"].iloc[-1])
    age_sec = float((pd.Timestamp.now("UTC") - last_ts).total_seconds())
    if age_sec > max_age_sec:
        return {
            "ts": last_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gate_on": 0,
            "gate_off": 1,
            "source": "stale_default_off",
            "age_sec": round(age_sec, 1),
            "error": f"renko_too_old:{age_sec:.1f}s",
        }

    chop_len = _env_int("LIVE_GATE_CHOP_LEN", 14)
    adx_len = _env_int("LIVE_GATE_ADX_LEN", 14)
    er_len = _env_int("LIVE_GATE_ER_LEN", 40)

    bars = bars.copy()
    bars["CHOP"] = _choppiness(bars, chop_len)
    bars["ADX"] = _adx(bars, adx_len)
    bars["ER"] = _efficiency_ratio(bars, er_len)

    # recent history used to fit dynamic thresholds
    fit_bars = _env_int("LIVE_GATE_FIT_BARS", 4000)
    hist = bars.tail(fit_bars).copy()

    req = ["CHOP", "ADX", "ER"]
    hist = hist.dropna(subset=req).reset_index(drop=True)
    if len(hist) < max(chop_len, adx_len, er_len, 100):
        return {
            "ts": last_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gate_on": 0,
            "gate_off": 1,
            "source": "default_off",
            "age_sec": round(age_sec, 1),
            "error": f"not_enough_history:{len(hist)}",
        }

    th = _fit_thresholds_live(hist)

    chop_ok, chop_bad = _hysteresis_high_is_good_last(hist["CHOP"], th["chop_on"], th["chop_off"])
    adx_ok, adx_bad = _hysteresis_low_is_good_last(hist["ADX"], th["adx_on"], th["adx_off"], start_on=True)
    er_ok, er_bad = _hysteresis_low_is_good_last(hist["ER"], th["er_on"], th["er_off"], start_on=True)

    # old confirmed logic = countertrend regime:
    # CHOP high + ADX low + ER low
    gate_countertrend_on = int((chop_ok + adx_ok + er_ok) >= 2)
    gate_trend_on = int(1 - gate_countertrend_on)

    primary = str(os.getenv("LIVE_GATE_PRIMARY", "off")).strip().lower()
    if primary in ("on", "countertrend", "flip"):
        gate_on = gate_countertrend_on
    else:
        gate_on = gate_trend_on

    gate_off = int(1 - gate_on)

    last = hist.iloc[-1]
    ts = pd.Timestamp(last["ts"]).strftime("%Y-%m-%dT%H:%M:%SZ")

    out: Dict[str, Any] = {
        "ts": ts,
        "gate_on": int(gate_on),
        "gate_off": int(gate_off),
        "source": "renko_live_chop_adx_er",
        "mode": str(os.getenv("LIVE_GATE_MODE", "rolling_quantiles")).strip().lower(),
        "primary": primary,
        "age_sec": round(age_sec, 1),

        # explicit regime views so executor/dashboard can choose cleanly later
        "gate_countertrend_on": int(gate_countertrend_on),
        "gate_trend_on": int(gate_trend_on),

        # latest indicator values
        "chop": round(float(last["CHOP"]), 4),
        "adx": round(float(last["ADX"]), 4),
        "er": round(float(last["ER"]), 6),

        # current metric states under old CHOP/ADX/ER semantics
        "chop_ok": int(chop_ok),
        "adx_ok": int(adx_ok),
        "er_ok": int(er_ok),
        "chop_bad": int(chop_bad),
        "adx_bad": int(adx_bad),
        "er_bad": int(er_bad),

        # active thresholds
        "chop_on_th": round(float(th["chop_on"]), 4) if np.isfinite(th["chop_on"]) else None,
        "chop_off_th": round(float(th["chop_off"]), 4) if np.isfinite(th["chop_off"]) else None,
        "adx_on_th": round(float(th["adx_on"]), 4) if np.isfinite(th["adx_on"]) else None,
        "adx_off_th": round(float(th["adx_off"]), 4) if np.isfinite(th["adx_off"]) else None,
        "er_on_th": round(float(th["er_on"]), 6) if np.isfinite(th["er_on"]) else None,
        "er_off_th": round(float(th["er_off"]), 6) if np.isfinite(th["er_off"]) else None,

        # fit config
        "fit_bars": int(fit_bars),
        "chop_len": int(chop_len),
        "adx_len": int(adx_len),
        "er_len": int(er_len),
    }

    if _env_bool("LIVE_GATE_DEBUG", False):
        out["debug_last_ts"] = str(last["ts"])

    _publish_gate_state_to_redis(out)

    return out